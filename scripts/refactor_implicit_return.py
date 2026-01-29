import os
import re
import sys

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Tokenize/Scan to find function boundaries
    # We need to handle strings, comments, and nesting to find the matching '}' for 'fn ... {'
    
    # States
    STATE_NORMAL = 0
    STATE_STRING = 1
    STATE_CHAR = 2
    STATE_LINE_COMMENT = 3
    STATE_BLOCK_COMMENT = 4
    
    state = STATE_NORMAL
    stack = [] # (type, start_index) where type is 'fn' or 'block'
    
    replacements = [] # (start, end, replacement_text)

    i = 0
    length = len(content)
    
    # We need to track if we just saw 'fn' before a '{'
    # Simple heuristic: searching backwards from '{' skipping white/comments to find 'fn'
    # But doing single pass is better.
    
    # Let's keep a buffer of recent tokens or just simplified "last_keyword" tracking
    # But 'fn foo() -> T {' can be long.
    
    # Alternative:
    # 1. Strip comments/strings for structural analysis (replace with spaces to keep indices)
    # 2. Find 'fn' keywords.
    
    clean_content = list(content)
    
    while i < length:
        char = content[i]
        
        if state == STATE_NORMAL:
            if char == '"':
                state = STATE_STRING
                clean_content[i] = ' '
            elif char == "'":
                state = STATE_CHAR
                clean_content[i] = ' '
            elif char == '/' and i + 1 < length and content[i+1] == '/':
                state = STATE_LINE_COMMENT
                clean_content[i] = ' '
                clean_content[i+1] = ' '
                i += 1
            elif char == '/' and i + 1 < length and content[i+1] == '*':
                state = STATE_BLOCK_COMMENT
                clean_content[i] = ' '
                clean_content[i+1] = ' '
                i += 1
            else:
                pass # structure kept
        elif state == STATE_STRING:
            clean_content[i] = ' '
            if char == '"' and content[i-1] != '\\':
                state = STATE_NORMAL
        elif state == STATE_CHAR:
            clean_content[i] = ' '
            if char == "'" and content[i-1] != '\\':
                state = STATE_NORMAL
        elif state == STATE_LINE_COMMENT:
            clean_content[i] = ' '
            if char == '\n':
                state = STATE_NORMAL
                clean_content[i] = '\n' # Keep newline
        elif state == STATE_BLOCK_COMMENT:
            clean_content[i] = ' '
            if char == '*' and i + 1 < length and content[i+1] == '/':
                state = STATE_NORMAL
                clean_content[i+1] = ' '
                i += 1
        
        i += 1
        
    structural_text = "".join(clean_content)
    
    # Now scan structural_text for braces and keywords
    # Regex to find 'fn' start or '{' or '}'
    
    # It's safer to just iterate tokens in structural_text
    
    token_iter = re.finditer(r'(fn\b)|(\{|\})|(;)', structural_text)
    
    brace_stack = [] # (is_function_body, start_index)
    last_keyword_fn = False
    
    # We need to distinguish "fn header { ... }" vs "if/loop { ... }"
    # structural_text has cleaned comments/strings.
    
    # We iterate tokens. If we see 'fn', we set flag. 
    # If we see '{', we check flag. Push to stack. Reset flag.
    # If we see ';', we reset flag (fn decl end or statement end).
    # Wait, 'fn foo() -> T;' is a declaration (no body). 'fn foo() -> T {' is definition.
    # But type param constraints 'where T: Copy {' can be between.
    
    # Better approach:
    # Use the token iterator.
    # Keep track of "latest 'fn' index seen before '{' without intervening ';'"
    
    latest_fn_idx = -1
    
    for match in token_iter:
        g = match.group(0)
        idx = match.start()
        
        if g == 'fn':
            latest_fn_idx = idx
        elif g == '{':
            # Check if this brace belongs to a function
            # Heuristic: if 'fn' was seen recently and no ';' or '}' or '{' intervened?
            # Actually, `fn foo() {` is common.
            # `impl T {` - NOT a function body.
            # `if ... {` - NOT.
            # `struct T {` - NOT.
            
            # The issue is distinguishing `impl ... { fn ...` vs `fn ... { ... }`.
            # Both have `fn` before `{`.
            # So if `latest_fn_idx` is valid (not reset), we assume it's a function body.
            
            is_func = False
            if latest_fn_idx != -1:
                # Make sure there wasn't a structure-breaking token between fn and {
                # We already iterate useful tokens. 
                # If we encounter another '{' or '}' or ';' between fn and {, then it wasn't that fn.
                # BUT, `fn foo() -> T {` has no `{` `}` `;` in between.
                # `fn foo<T, U>() where T: A {` also fine.
                is_func = True
                latest_fn_idx = -1 # Consumed
            
            brace_stack.append({'is_func': is_func, 'open_idx': idx})
            
            # Reset fn index for nested scopes (nested fn is rare but possible, logic safe)
            latest_fn_idx = -1 
            
        elif g == '}':
            if not brace_stack:
                continue # Unbalanced?
            
            block = brace_stack.pop()
            if block['is_func']:
                # found end of function
                close_idx = idx
                open_idx = block['open_idx']
                
                # Check the content immediately before close_idx
                # Look at `structural_text` backwards from close_idx
                
                # Scan backwards skipping whitespace/comments (already spaces in structural_text)
                scan_pos = close_idx - 1
                while scan_pos > open_idx and structural_text[scan_pos].isspace():
                    scan_pos -= 1
                
                # Expect ';' at scan_pos
                if structural_text[scan_pos] == ';':
                    semi_col_idx = scan_pos
                    # Now find matching 'return'
                    # We need to ensure it's "return <expr> ;"
                    # Scan backwards from semi_col_idx to find 'return'
                    # BUT carefully: `return 1 + { return 2; };` -> complex expression.
                    # We only care if the statement STARTS with return.
                    
                    # Instead of parsing expression backwards, let's look at the "Last Statement" in this block.
                    # We can find the previous `;` or `{` or `}` to bound the statement.
                    
                    stmt_end = semi_col_idx
                    # Search backwards for statement boundary
                    # Boundary: `{`, `}`, `;`
                    
                    bound_pos = stmt_end - 1
                    nest_depth = 0
                    found_bound = False
                    
                    while bound_pos > open_idx:
                        c = structural_text[bound_pos]
                        if c == '}' or c == ')' or c == ']':
                            nest_depth += 1
                        elif c == '{' or c == '(' or c == '[':
                            if nest_depth > 0:
                                nest_depth -= 1
                            else:
                                # Hit start of block -> boundary found
                                found_bound = True
                                break
                        elif c == ';':
                            if nest_depth == 0:
                                found_bound = True
                                break
                        bound_pos -= 1
                    
                    stmt_start = bound_pos + 1
                    
                    # Extract statement text (from original content) to verify 'return'
                    # Use structural_text for checking keywords to avoid string hits
                    
                    stmt_struct = structural_text[stmt_start:stmt_end].strip()
                    
                    if stmt_struct.startswith("return"):
                        # Verify it is indeed "return" token
                        # e.g. "return5" is var? NO, lexer. "return" must be followed by space or non-ident.
                        # Check original content at that pos
                        
                        # Locate "return" in original content
                        # We know relative pos in stmt
                        
                        # Find literal "return" in that range
                        # stmt_text may contain comments/spaces.
                        
                        # Regex search in the identified range
                        stmt_range_content = content[stmt_start:stmt_end]
                        # Look for `^\s*return\b`
                        
                        m = re.match(r'^\s*return\b', stmt_range_content)
                        if m:
                            # Found it!
                            # We want to replace "return <expr> ;" with "<expr>"
                            # The range to modify is [stmt_start + match.end(), semi_col_idx + 1] ? 
                            # No, we remove "return" (and following space) and the trailing ";".
                            pass
                            
                            # Calculate absolute removal ranges
                            # 1. "return" keyword removal
                            # stmt_start is absolute index.
                            match_start = stmt_start + m.start()
                            match_end = stmt_start + m.end()
                            
                            # 2. Semicolon removal
                            # semi_col_idx is absolute index.
                            
                            # Add to replacements
                            # Be careful: We want to perform all replacements for a file.
                            # Replacements should be: (start, end, replacement)
                            # "return " -> "" (at start)
                            # ";" -> "" (at end)
                            
                            # Check if return is "return;" (void return) -> remove line? 
                            # If "return;", expr is empty. Implicit return of ()?
                            # tl supports implicit return unit? Yes, empty block returns unit.
                            # So `return;` -> ``. 
                            
                            replacements.append((match_start, match_end, "")) 
                            replacements.append((semi_col_idx, semi_col_idx + 1, ""))
                
        elif g == ';':
            latest_fn_idx = -1 # Definition ended or structural separation
            
    # Apply replacements from end to start
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    if not replacements:
        return False

    new_content = list(content)
    for start, end, text in replacements:
        # e.g. replace range
        new_content[start:end] = list(text)
        
    final_text = "".join(new_content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_text)
        
    print(f"Refactored: {filepath}")
    return True

if __name__ == '__main__':
    # Find all .tl files
    root_dir = os.getcwd()
    tl_files = []
    for root, dirs, files in os.walk(root_dir):
        # Exclude hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        dirs[:] = [d for d in dirs if d != 'target']
        
        for file in files:
            if file.endswith('.tl'):
                tl_files.append(os.path.join(root, file))
    
    count = 0
    for f in tl_files:
        if process_file(f):
            count += 1
            
    print(f"Total files refactored: {count}")
