import re
import sys

def analyze(filename):
    # Track which pointers are currently "live" (allocated but not freed)
    live = {}   # ptr -> {'loc': ..., 'line': ...}
    
    # Match both normal ALLOC and GradRecycle ALLOC
    alloc_pattern = re.compile(r'\[ALLOC\](?:\s*\(GradRecycle\))?\s*Ptr: (0x[0-9a-fA-F]+) at (.*)')
    free_pattern = re.compile(r'\[FREE\] Ptr: (0x[0-9a-fA-F]+) at (.*)')
    
    alloc_count = 0
    free_count = 0
    double_free_count = 0
    realloc_without_free = 0

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            m_alloc = alloc_pattern.search(line)
            if m_alloc:
                ptr = m_alloc.group(1)
                loc = m_alloc.group(2)
                alloc_count += 1
                if ptr in live:
                    realloc_without_free += 1
                live[ptr] = {'loc': loc, 'line': line_num}
                continue
                
            m_free = free_pattern.search(line)
            if m_free:
                ptr = m_free.group(1)
                loc = m_free.group(2)
                free_count += 1
                if ptr in live:
                    del live[ptr]
                else:
                    double_free_count += 1
                continue

    print(f"=== メモリリーク解析結果 ===")
    print(f"Total ALLOCs:              {alloc_count}")
    print(f"Total FREEs:               {free_count}")
    print(f"再確保 (free 前の上書き):    {realloc_without_free}")
    print(f"二重解放/不明解放:          {double_free_count}")
    print(f"")
    print(f"未解放 (確保のみ):          {len(live)}")
    
    if live:
        print(f"\n未解放ポインタ (先頭20件):")
        for ptr, info in list(live.items())[:20]:
            print(f"  {ptr} allocated at {info['loc']} (log line {info['line']})")
        if len(live) > 20:
            print(f"  ... 他 {len(live) - 20} 件")
    else:
        print("\nリークなし ✅")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_leaks.py <log_file>")
        sys.exit(1)
    analyze(sys.argv[1])
