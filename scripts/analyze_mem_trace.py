
import re
import sys

def parse_log(filename):
    allocs = {}  # ptr -> (file, line, size, count)
    
    # Regex patterns
    # [ALLOC] File: ..., Line: ..., Size: ..., Ptr: 0x...
    alloc_pattern = re.compile(r'\[ALLOC\] File: (.+), Line: (\d+), Size: (\d+), Ptr: (0x[0-9a-fA-F]+)')
    # [FREE] File: ..., Line: ..., Ptr: 0x...
    free_pattern = re.compile(r'\[FREE\] File: (.+), Line: (\d+), Ptr: (0x[0-9a-fA-F]+)')

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check Alloc
            m_alloc = alloc_pattern.search(line)
            if m_alloc:
                file = m_alloc.group(1)
                line_num = int(m_alloc.group(2))
                size = int(m_alloc.group(3))
                ptr = m_alloc.group(4)
                
                if ptr in allocs:
                    # Reused address without free? It happens if we missed a Free log, 
                    # or if address space reused quickly.
                    # Ideally we treat it as a new alloc, validating strict LIFO/matching is hard without unique IDs.
                    # But for now, let's assume valid pairs. 
                    # If we overwrite, we lose the previous leak info if it WAS a leak.
                    # So we print a warning? 
                    # Actually, if we alloc again on same ptr, previous one IS leaked if not freed!
                    # BUT, malloc may return same address if previous was freed.
                    # So if it is in allocs, it means we think it is still alive. So it's a double alloc (impossible) or we missed the free.
                    pass 
                
                allocs[ptr] = (file, line_num, size)
                continue

            # Check Free
            m_free = free_pattern.search(line)
            if m_free:
                file = m_free.group(1)
                line_num = int(m_free.group(2))
                ptr = m_free.group(3)
                
                if ptr in allocs:
                    del allocs[ptr]
                else:
                    # Double free or freeing something not tracked (e.g. internal runtime allocs)
                    pass
    
    return allocs

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_mem_trace.py <log_file>")
        sys.exit(1)
        
    filename = sys.argv[1]
    leaks = parse_log(filename)
    
    total_leaked_bytes = 0
    print(f"--- Leaked Allocations ({len(leaks)}) ---")
    
    # Group by location
    grouped = {}
    
    for ptr, (file, line, size) in leaks.items():
        total_leaked_bytes += size
        key = f"{file}:{line}"
        if key not in grouped:
            grouped[key] = {'count': 0, 'size': 0}
        grouped[key]['count'] += 1
        grouped[key]['size'] += size

    # Sort by size desc
    sorted_leaks = sorted(grouped.items(), key=lambda x: x[1]['size'], reverse=True)
    
    for loc, data in sorted_leaks:
        print(f"{loc} => Count: {data['count']}, Total Size: {data['size']} bytes")
        
    print("--------------------------------")
    print(f"Total Leaked Bytes: {total_leaked_bytes}")

if __name__ == "__main__":
    main()
