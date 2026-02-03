use std::collections::HashMap;
use crate::compiler::ast::*;

#[derive(Debug, Default)]
pub struct FunctionAnalysis {
    pub slots: HashMap<String, usize>, // Legacy: For Allocator (Variable -> Slot ID)
    pub num_slots: usize,
    pub last_use_times: HashMap<usize, usize>, // DefTime -> LastUseTime
}

pub struct LivenessAnalyzer {
    // DefTime -> LastUseTime
    last_use_times: HashMap<usize, usize>,
    
    // Scope Management: Name -> DefTime
    scopes: Vec<HashMap<String, usize>>,
    
    // Current instruction index (virtual time)
    current_time: usize,
    
    // Legacy: Slot Map (Name -> Slot) - Rough approx for now
    slot_map: HashMap<String, usize>,
}

impl LivenessAnalyzer {
    pub fn new() -> Self {
        LivenessAnalyzer {
            last_use_times: HashMap::new(),
            scopes: vec![HashMap::new()],
            current_time: 0,
            slot_map: HashMap::new(),
        }
    }

    pub fn analyze(func: &FunctionDef) -> FunctionAnalysis {
        let mut analyzer = LivenessAnalyzer::new();
        analyzer.visit_function(func);
        // analyzer.assign_slots(); // Legacy: Slot assignment disabled for now
        
        // let num_slots = analyzer.calculate_max_slots();
        let num_slots = 0;
        
        FunctionAnalysis {
            slots: analyzer.slot_map, // Will be empty
            num_slots,
            last_use_times: analyzer.last_use_times,
        }

    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn visit_function(&mut self, func: &FunctionDef) {
        // Arguments are defined at time 0
        for (arg_name, _) in &func.args {
            self.define_var(arg_name, 0);
        }

        for stmt in &func.body {
            self.visit_stmt(stmt);
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        self.current_time += 1;
        let time = self.current_time;

        match &stmt.inner {
            StmtKind::Let { name, value, .. } => {
                self.visit_expr(value); // RHS used first
                self.define_var(name, time);
            }
            StmtKind::Assign { name, value, .. } => {
                self.visit_expr(value);
                self.use_var(name, time);
            }
            StmtKind::TensorDecl { name, init, .. } => {
                 if let Some(expr) = init {
                     self.visit_expr(expr);
                 }
                 self.define_var(name, time);
            }
            StmtKind::Expr(e) => self.visit_expr(e),
            StmtKind::Return(e) => {
                if let Some(expr) = e {
                    self.visit_expr(expr);
                }
            }
            StmtKind::For { loop_var, iterator, body } => {
                self.enter_scope(); // Loop Scope
                
                self.visit_expr(iterator);
                // Loop var defined
                self.define_var(loop_var, time);
                
                // Loop body execution
                let loop_start = time;
                for s in body {
                    self.visit_stmt(s);
                }
                let loop_end = self.current_time;
                self.extend_loop_vars(loop_start, loop_end);
                
                self.exit_scope();
            }
            StmtKind::While { cond, body } => {
                self.visit_expr(cond);
                let loop_start = time;
                for s in body {
                    self.visit_stmt(s);
                }
                let loop_end = self.current_time;
                self.extend_loop_vars(loop_start, loop_end);
            }
            // ... Other stmts
            _ => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        self.current_time += 1;
        let time = self.current_time;
        match &expr.inner {
            ExprKind::Variable(name) => {
                self.use_var(name, time);
            }
            ExprKind::BinOp(l, _, r) => {
                self.visit_expr(l);
                self.visit_expr(r);
            }
            ExprKind::UnOp(_, e) => self.visit_expr(e),
            ExprKind::FnCall(_, args) => {
                for arg in args { self.visit_expr(arg); }
            }
             ExprKind::MethodCall(obj, _, args) => {
                self.visit_expr(obj);
                for arg in args { self.visit_expr(arg); }
            }
            ExprKind::Tuple(exprs)
            | ExprKind::TensorLiteral(exprs)
            | ExprKind::TensorConstLiteral(exprs) => {
                for e in exprs {
                    self.visit_expr(e);
                }
            }
            ExprKind::Range(start, end) => {
                self.visit_expr(start);
                self.visit_expr(end);
            }
            ExprKind::IndexAccess(val, idxs) => {
                self.visit_expr(val);
                for idx in idxs {
                    self.visit_expr(idx);
                }
            }
            ExprKind::TupleAccess(val, _) | ExprKind::FieldAccess(val, _) => {
                self.visit_expr(val);
            }
            ExprKind::As(val, _) => {
                self.visit_expr(val);
            }
            ExprKind::IfExpr(cond, then_block, else_block_opt) => {
                self.visit_expr(cond);
                
                self.enter_scope();
                for s in then_block {
                    self.visit_stmt(s);
                }
                self.exit_scope();

                if let Some(else_block) = else_block_opt {
                    self.enter_scope();
                    for s in else_block {
                        self.visit_stmt(s);
                    }
                    self.exit_scope();
                }
            }
            ExprKind::Block(stmts) => {
                self.enter_scope();
                for s in stmts {
                    self.visit_stmt(s);
                }
                self.exit_scope();
            }
            ExprKind::IfLet { expr, then_block, else_block, .. } => {
                self.visit_expr(expr);
                
                self.enter_scope();
                for s in then_block {
                    self.visit_stmt(s);
                }
                self.exit_scope();
                
                if let Some(else_block) = else_block {
                    self.enter_scope();
                    for s in else_block {
                        self.visit_stmt(s);
                    }
                    self.exit_scope();
                }
            }
            ExprKind::Match { expr, arms } => {
                self.visit_expr(expr);
                for (_, body_expr) in arms {
                    self.enter_scope();
                    self.visit_expr(body_expr);
                    self.exit_scope();
                }
            }
            ExprKind::StructInit(_, fields) => {
                for (_, expr) in fields {
                    self.visit_expr(expr);
                }
            }
            ExprKind::EnumInit { payload, .. } => {
                match payload {
                    crate::compiler::ast::EnumVariantInit::Unit => {},
                    crate::compiler::ast::EnumVariantInit::Tuple(exprs) => {
                        for e in exprs {
                            self.visit_expr(e);
                        }
                    },
                    crate::compiler::ast::EnumVariantInit::Struct(fields) => {
                        for (_, expr) in fields {
                            self.visit_expr(expr);
                        }
                    }
                }
            }
            ExprKind::TensorComprehension { clauses, body, .. } => {
                self.enter_scope();
                for c in clauses {
                    match c {
                        ComprehensionClause::Generator { range, .. } => self.visit_expr(range), // Range evaluated in OUTER scope? No, logic varies.
                        ComprehensionClause::Condition(cond) => self.visit_expr(cond), // Condition vars?
                    }
                }
                if let Some(b) = body {
                    self.visit_expr(b);
                }
                self.exit_scope();
            }
            // Literals that don't contain other expressions
            ExprKind::Float(_)
            | ExprKind::Int(_)
            | ExprKind::Bool(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::CharLiteral(_)
            | ExprKind::Symbol(_)
            | ExprKind::LogicVar(_)
            | ExprKind::Wildcard => {}

            // Handle StaticMethodCall separately if needed, passing to FnCall logic?
            ExprKind::StaticMethodCall(_, _, args) => {
                for arg in args {
                    self.visit_expr(arg);
                }
            }

        }
    }

    fn define_var(&mut self, name: &str, time: usize) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), time);
        }
        // Initialize last use as definition time (zero length life initially)
        self.last_use_times.insert(time, time);
        
        // Legacy Map (Approximate)
        if !self.slot_map.contains_key(name) {
             let slot = self.slot_map.len();
             self.slot_map.insert(name.to_string(), slot);
        }
    }

    fn use_var(&mut self, name: &str, time: usize) {
        // Find definition in nearest scope
        for scope in self.scopes.iter().rev() {
            if let Some(&def_time) = scope.get(name) {
                if let Some(last_use) = self.last_use_times.get_mut(&def_time) {
                    *last_use = std::cmp::max(*last_use, time);
                }
                return;
            }
        }
    }
    
    // Naive loop handling: If a var's interval overlaps with loop, extend to end of loop?
    // No, only if it is used IN the loop.
    // We need to know which variables were touched effectively.
    // For now, let's keep it simple: Just linear scan.
    // The issue with linear scan for loops: use at global time T+10 (inside loop) implies it's live at T (loop start) if defined before.
    // But our linear scan handles that: definition was at T-5, usage at T+10 -> Interval is [T-5, T+10].
    // What about back-edge? i is used in condition (T) and incremented at end (T+10).
    // Start T, End T+10. Matches reality. 
    // The only issue is if something is defined inside loop and carried over? 
    // e.g. x = ... (T+2); (next iter) use x (T+1). But T+1 < T+2 ??
    // Re-visiting order matters for intra-loop dependencies.
    // Conservative: Treat loop body as opaque block where everything used inside extends to block end?
    fn extend_loop_vars(&mut self, _start: usize, _end: usize) {
        // Warning: Correct loop handling requires full dataflow or iterative scanning.
        // For prototype, we assume code is mostly straight-line or simple loops.
        // Or we revisit loop body?
    }

    /*
    fn assign_slots(&mut self) {
        // Disabled
    }

    fn calculate_max_slots(&self) -> usize {
        0
    }
    */

}
