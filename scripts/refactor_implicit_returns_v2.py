import os
import re

class SourceScanner:
    def __init__(self, content):
        self.content = content
        self.length = len(content)
        self.pos = 0
        self.replacements = [] # list of (start, end, text)

    def scan(self):
        depth = 0
        fn_depth_stack = [] 
        scope_candidates = {}
        stmt_start_positions = {0: 0}

        def enter_block():
            nonlocal depth
            depth += 1
            stmt_start_positions[depth] = self.pos + 1
            scope_candidates[depth] = None

        def exit_block(end_pos):
            nonlocal depth
            
            is_fn_end = False
            if fn_depth_stack and fn_depth_stack[-1] == depth - 1:
                is_fn_end = True
            
            if is_fn_end:
                cand = scope_candidates.get(depth)
                stmt_start = stmt_start_positions.get(depth, end_pos)
                tail_text = self.content[stmt_start:end_pos].strip()
                
                if tail_text:
                    m = re.match(r'^return\s+([^;]+)$', tail_text)
                    if m:
                        expr = m.group(1).strip()
                        # Preserve indentation if we can, but tail_text is stripped.
                        # We are replacing (stmt_start, end_pos).
                        self.replacements.append((stmt_start, end_pos, f" {expr} "))
                        cand = None 
                
                if cand:
                    r_start, r_end = cand
                    full_stmt = self.content[r_start:r_end]
                    # Regex to extract expr, handling leading whitespace
                    m = re.match(r'^(\s*)return\s+(.*);', full_stmt, re.DOTALL)
                    if m:
                        indent = m.group(1)
                        expr = m.group(2).strip()
                        self.replacements.append((r_start, r_end, f"{indent}{expr}"))
                        
                if is_fn_end:
                    fn_depth_stack.pop()

            depth -= 1

        i = 0
        while i < self.length:
            self.pos = i
            c = self.content[i]
            
            if c == '"':
                i += 1
                while i < self.length:
                    if self.content[i] == '\\':
                        i += 2
                        continue
                    if self.content[i] == '"':
                        break
                    i += 1
                i += 1
                if depth in scope_candidates and scope_candidates[depth]:
                     scope_candidates[depth] = None
                continue
                
            if c == "'":
                i += 1
                while i < self.length:
                    if self.content[i] == '\\':
                        i += 2
                        continue
                    if self.content[i] == "'":
                        break
                    i += 1
                i += 1
                if depth in scope_candidates and scope_candidates[depth]:
                     scope_candidates[depth] = None
                continue
                
            if c == '/' and i+1 < self.length:
                if self.content[i+1] == '/':
                    i += 2
                    while i < self.length and self.content[i] != '\n':
                        i += 1
                    i += 1
                    continue
                elif self.content[i+1] == '*':
                    i += 2
                    while i+1 < self.length:
                        if self.content[i] == '*' and self.content[i+1] == '/':
                            i += 2
                            break
                        i += 1
                    continue

            if c == 'f' and self.content[i:i+3] == 'fn ':
                fn_depth_stack.append(depth)
                pass

            if c == '{':
                enter_block()
                i += 1
                continue
                
            if c == '}':
                exit_block(i)
                i += 1
                continue
            
            if c == ';':
                start = stmt_start_positions.get(depth, 0)
                end = i + 1
                stmt = self.content[start:end].strip()
                
                if stmt.startswith('return') and not stmt.startswith('return;'):
                     if re.match(r'return\s+', stmt) or stmt == 'return':
                         scope_candidates[depth] = (start, end)
                else:
                    scope_candidates[depth] = None
                
                stmt_start_positions[depth] = i + 1
                i += 1
                continue

            if c.isspace():
                i += 1
                continue
            
            if depth in scope_candidates and scope_candidates[depth]:
                 scope_candidates[depth] = None
            
            i += 1
            
    def get_refactored(self):
        self.scan()
        if not self.replacements:
            return None
        
        self.replacements.sort(key=lambda x: x[0], reverse=True)
        new_content = self.content
        for start, end, text in self.replacements:
            new_content = new_content[:start] + text + new_content[end:]
        return new_content

def refactor_file_v2(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    scanner = SourceScanner(content)
    new_content = scanner.get_refactored()
    
    if new_content and new_content != content:
        print(f"Refactoring {filepath}")
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    roots = ['tests', 'examples', 'src/compiler/codegen/builtin_types']
    base_dir = os.getcwd()
    
    count = 0
    for root in roots:
        full_root = os.path.join(base_dir, root)
        if not os.path.exists(full_root):
            continue
            
        for dirpath, dirnames, filenames in os.walk(full_root):
            for filename in filenames:
                if filename.endswith('.tl'):
                    filepath = os.path.join(dirpath, filename)
                    if refactor_file_v2(filepath):
                        count += 1
    print(f"Refactored {count} files.")

if __name__ == "__main__":
    main()
