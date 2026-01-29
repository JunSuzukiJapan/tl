import os
import re

def refactor_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Regex to find return statements at the end of a block
    # Matches: return <expr>; whitespace }
    # Group 1: <expr> (lazy)
    # Group 2: whitespace ending with }
    pattern = r'return\s*(.*?);\s*(\})'
    
    # Replacement function to handle empty return (Void) vs value return
    def replace(match):
        expr = match.group(1).strip()
        end = match.group(2)
        
        if not expr:
            # return; -> remove entirely (just keep the closing brace and preceding whitespace/newline if kept in group 2?)
            # Wait, group 2 consumes the whitespace before }.
            # If we want to remove 'return;', we effectively replace match with '}'.
            # But we might lose newline.
            # match is 'return; \n}'
            # match.group(2) is ' \n}'
            # We want just '\n}' ?
            return end
        else:
            # return 1; -> 1
            # We want to keep the whitespace found in group 2 (it includes the })
            # But the 'return' keyword was consumed.
            # implicit return is just <expr>
            return f"{expr}{end}"

    # Actually, simplistic regex substitution:
    # re.sub(pattern, replacement, content)
    # But checking for empty expr is safer manually or via lambda.
    
    # Refined pattern:
    # Note: explicit [^;]* to capture expression until semicolon
    # Handles `return;` (empty group 1) and `return 1;`
    regex = r'return\s*([^;]*?);\s*(\})'
    
    new_content = re.sub(regex, replace, content)
    
    if new_content != content:
        print(f"Refactoring {filepath}")
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    # Root directories to search
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
                    if refactor_file(filepath):
                        count += 1
                        
    print(f"Refactored {count} files.")

if __name__ == '__main__':
    main()
