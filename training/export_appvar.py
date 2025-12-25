#!/usr/bin/env python3
"""
Convert binary weights to TI-84 Plus CE AppVar format (.8xv).

Based on the TI-83/84 file format specification.
"""

import argparse
import struct
from pathlib import Path


def create_appvar(name: str, data: bytes) -> bytes:
    """
    Create a TI-84 Plus CE AppVar (.8xv) file.

    Args:
        name: AppVar name (max 8 characters, uppercase)
        data: Binary data to store

    Returns:
        Complete .8xv file contents
    """
    name = name.upper()[:8]

    # File header
    signature = b'**TI83F*'
    sig_extra = bytes([0x1A, 0x0A, 0x00])
    comment = b'Created by CalculatorLLM'
    comment = comment[:42].ljust(42, b'\x00')

    # Variable data = 2-byte size prefix + actual data
    var_data = struct.pack('<H', len(data)) + data
    var_data_len = len(var_data)

    # Name padded to 8 bytes
    name_bytes = name.encode('ascii').ljust(8, b'\x00')

    # Build variable entry
    var_entry = b''
    var_entry += struct.pack('<H', var_data_len + 2)  # Size for flash
    var_entry += struct.pack('<H', var_data_len)      # Size of var_data
    var_entry += bytes([0x15])                         # Type: AppVar
    var_entry += name_bytes                            # Name (8 bytes)
    var_entry += bytes([0x00])                         # Version
    var_entry += bytes([0x00])                         # Flag: not archived
    var_entry += struct.pack('<H', var_data_len)      # Size again
    var_entry += var_data                              # Data

    data_section_len = len(var_entry)
    checksum = sum(var_entry) & 0xFFFF

    output = b''
    output += signature
    output += sig_extra
    output += comment
    output += struct.pack('<H', data_section_len)
    output += var_entry
    output += struct.pack('<H', checksum)

    return output


def main():
    parser = argparse.ArgumentParser(description='Convert weights to TI-84 AppVar')
    parser.add_argument('input', type=str, help='Input binary weights file')
    parser.add_argument('output', type=str, nargs='?', default='GRUMODEL.8xv',
                        help='Output AppVar file (default: GRUMODEL.8xv)')
    parser.add_argument('--name', type=str, default='GRUMODEL',
                        help='AppVar name (max 8 chars, default: GRUMODEL)')
    args = parser.parse_args()

    # Read input weights
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    data = input_path.read_bytes()
    print(f"Read {len(data):,} bytes from {input_path}")

    # Create AppVar
    appvar = create_appvar(args.name, data)
    print(f"Created AppVar '{args.name}' ({len(appvar):,} bytes total)")

    # Write output
    output_path = Path(args.output)
    output_path.write_bytes(appvar)
    print(f"Saved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
