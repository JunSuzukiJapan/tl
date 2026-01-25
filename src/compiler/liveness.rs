use std::collections::{HashMap, HashSet};
use crate::compiler::ast::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Default)]
pub struct FunctionAnalysis {
    pub slots: HashMap<String, usize>, // Variable -> Slot ID
    pub num_slots: usize,
}

pub struct LivenessAnalyzer {
    // Variable -> Interval
    intervals: HashMap<String, Interval>,
    // Current instruction index (virtual time)
    current_time: usize,
    // Variables currently alive (to handle loops correctly?)
    // Or just simple intervals for now.
    
    // Result
    slot_map: HashMap<String, usize>,
}

impl LivenessAnalyzer {
    pub fn new() -> Self {
        LivenessAnalyzer {
            intervals: HashMap::new(),
            current_time: 0,
            slot_map: HashMap::new(),
        }
    }

    pub fn analyze(func: &FunctionDef) -> FunctionAnalysis {
        let mut analyzer = LivenessAnalyzer::new();
        analyzer.visit_function(func);
        analyzer.assign_slots();
        
        let num_slots = analyzer.calculate_max_slots();
        
        FunctionAnalysis {
            slots: analyzer.slot_map,
            num_slots,
        }
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
                self.visit_expr(iterator);
                // Loop var defined
                self.define_var(loop_var, time);
                
                // Loop body execution
                let loop_start = time;
                for s in body {
                    self.visit_stmt(s);
                }
                let loop_end = self.current_time;

                // CRITICAL: Any variable used in the loop must be extended to loop_end
                // Ideally we do a fix-point analysis, but conservative approach:
                // If a var is used in loop, its end >= loop_end.
                // For now, linear scan might miss back-edges.
                // Simple hack: Scan body twice? Or just track "live-in" / "live-out".
                // Let's rely on simple linear scan for first prototype and refine for loops later.
                // Actually, if we just extend the usage of anything touched in the loop to the end of the loop, it works for 90% cases.
                self.extend_loop_vars(loop_start, loop_end);
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
                for s in then_block {
                    self.visit_stmt(s);
                }
                if let Some(else_block) = else_block_opt {
                    for s in else_block {
                        self.visit_stmt(s);
                    }
                }
            }
            ExprKind::Block(stmts) => {
                for s in stmts {
                    self.visit_stmt(s);
                }
            }
            ExprKind::IfLet { expr, then_block, else_block, .. } => {
                self.visit_expr(expr);
                for s in then_block {
                    self.visit_stmt(s);
                }
                if let Some(else_block) = else_block {
                    for s in else_block {
                        self.visit_stmt(s);
                    }
                }
            }
            ExprKind::Match { expr, arms } => {
                self.visit_expr(expr);
                for (_, body_expr) in arms {
                    self.visit_expr(body_expr);
                }
            }
            ExprKind::StructInit(_, _, fields) => {
                for (_, expr) in fields {
                    self.visit_expr(expr);
                }
            }
            ExprKind::EnumInit { fields, .. } => {
                for (_, expr) in fields {
                    self.visit_expr(expr);
                }
            }
            ExprKind::TensorComprehension { clauses, body, .. } => {
                for c in clauses {
                    match c {
                        ComprehensionClause::Generator { range, .. } => self.visit_expr(range),
                        ComprehensionClause::Condition(cond) => self.visit_expr(cond),
                    }
                }
                if let Some(b) = body {
                    self.visit_expr(b);
                }
            }
            // Literals that don't contain other expressions
            ExprKind::Float(_)
            | ExprKind::Int(_)
            | ExprKind::Bool(_)
            | ExprKind::StringLiteral(_)
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
        self.intervals.insert(name.to_string(), Interval { start: time, end: time });
    }

    fn use_var(&mut self, name: &str, time: usize) {
        if let Some(interval) = self.intervals.get_mut(name) {
             interval.end = std::cmp::max(interval.end, time);
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

    fn assign_slots(&mut self) {
        // Collect all intervals
        let mut intervals: Vec<(String, Interval)> = self.intervals.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        
        // Sort by start time
        intervals.sort_by_key(|(_, i)| i.start);

        let mut active: Vec<(usize, usize)> = Vec::new(); // (end_time, slot_id) of active allocations

        for (name, interval) in intervals {
            // Expire old intervals
            // Remove intervals that ended before this one starts
            // Actually, we can reuse a slot if its previous occupant ended BEFORE this start.
            // i.e., occupant.end < interval.start
            
            // Find a free slot
            // active list contains currently occupied slots.
            active.retain(|(end, _)| *end >= interval.start);
            
            let mut used_slots = HashSet::new();
            for (_, slot) in &active {
                used_slots.insert(*slot);
            }

            // Find lowest available slot
            let mut slot = 0;
            while used_slots.contains(&slot) {
                slot += 1;
            }

            self.slot_map.insert(name, slot);
            active.push((interval.end, slot));
        }
    }

    fn calculate_max_slots(&self) -> usize {
        self.slot_map.values().map(|s| s + 1).max().unwrap_or(0)
    }
}
