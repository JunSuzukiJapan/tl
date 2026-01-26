
import sys
import re

def analyze(filename):
    # Track RefCounts: ptr -> count
    refcounts = {}
    # Track events: list of (line_num, message)
    history = []
    
    # Regex patterns
    # [TL_MEM] register_struct ptr=0x...
    # [TL_MEM] register_tensor ptr=0x...
    # [TL_MEM] acquire ptr=0x... refcount=...
    # [TL_MEM] release ptr=0x... refcount=...
    # [TL_MEM] unregister ptr=0x...
    
    p_ptr = re.compile(r'ptr=(0x[0-9a-fA-F]+)')
    p_rc = re.compile(r'refcount=(\d+)')
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line.startswith("[TL_MEM]"):
                continue
            
            ptr_match = p_ptr.search(line)
            if not ptr_match:
                continue
                
            ptr = ptr_match.group(1)
            
            # Parse action
            if "register_" in line:
                # Register sets refcount to 1 (new) or increments (existing)
                # But our log says "refcount=X".
                # If "is_new", set to 1.
                # Use the logged refcount?
                rc_match = p_rc.search(line)
                if rc_match:
                    refcounts[ptr] = int(rc_match.group(1))
                else:
                    # struct register logs don't always show refcount newly?
                    # "register_struct ptr=... depth=..."
                    # We assume it becomes 1?
                    if ptr not in refcounts:
                        refcounts[ptr] = 1
                    else:
                        refcounts[ptr] += 1
            
            elif "acquire" in line:
                if ptr not in refcounts:
                    print(f"Line {i}: Acquire on unknown ptr {ptr}")
                    refcounts[ptr] = 1
                else:
                    if rc_match := p_rc.search(line):
                         # Trust log
                         refcounts[ptr] = int(rc_match.group(1))
                    else:
                        refcounts[ptr] += 1

            elif "release" in line:
                if ptr not in refcounts:
                    print(f"Line {i}: Release on unknown/freed ptr {ptr}")
                    # Potential Double Free if we track freed!
                else:
                    if rc_match := p_rc.search(line):
                        rc = int(rc_match.group(1))
                        refcounts[ptr] = rc
                        if rc == 0:
                            del refcounts[ptr]
                            print(f"Line {i}: Freed {ptr}")
                    else:
                        # Assume release decrements
                        refcounts[ptr] -= 1
                        if refcounts[ptr] <= 0:
                            del refcounts[ptr]
            
            elif "unregister" in line:
                # Unregister removes ownership but doesn't free.
                # However, our MemoryManager decrement logic isn't triggered.
                # It just removes from scope.
                # Log shows refcount BEFORE removal? Or generic?
                # "unregister ptr=... refcount=..."
                pass
                
    print("Analysis complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze.py <log_file>")
        sys.exit(1)
    analyze(sys.argv[1])
