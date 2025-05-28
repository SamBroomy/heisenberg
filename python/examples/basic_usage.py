#!/usr/bin/env python3

import heisenberg


def main():
    # Initialize Heisenberg
    hb = heisenberg.Heisenberg(overwrite_indexes=False)

    # Example search
    results = hb.search(["New York", "California"])
    print(f"Found {len(results)} results")

    # Example admin search
    admin_result = hb.admin_search("United States", levels=[0])
    if admin_result is not None:
        print(f"Admin search returned DataFrame with {admin_result.height} rows")

    # Example location resolution
    resolved = hb.resolve_location(["San Francisco", "CA", "USA"])
    for result in resolved:
        print(f"Resolved: {result}")


if __name__ == "__main__":
    main()
