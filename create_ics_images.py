#!/usr/bin/env python3
"""
Create visual diagrams for ICS course notes
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

# Set up Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
output_dir = r"D:\github\stibiums.github.io\assets\img\notes_img\ICS"
os.makedirs(output_dir, exist_ok=True)

def create_number_conversion():
    """Create number system conversion diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Number to convert
    decimal_num = 15213
    binary_num = bin(decimal_num)[2:]  # Remove '0b' prefix
    hex_num = hex(decimal_num)[2:].upper()  # Remove '0x' prefix
    
    # Create conversion flow
    systems = [
        ('十进制\nDecimal', str(decimal_num), '#FFE4B5'),
        ('二进制\nBinary', binary_num, '#E0FFFF'),
        ('十六进制\nHexadecimal', hex_num, '#F0E68C')
    ]
    
    # Position systems in triangle
    positions = [(0.5, 0.8), (0.2, 0.3), (0.8, 0.3)]
    
    for i, ((title, value, color), (x, y)) in enumerate(zip(systems, positions)):
        # Draw boxes
        box = FancyBboxPatch((x-0.15, y-0.1), 0.3, 0.2, 
                           boxstyle="round,pad=0.02", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=2)
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y+0.05, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x, y-0.05, value, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='red')
    ax.annotate('', xy=(0.35, 0.4), xytext=(0.35, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.4), xytext=(0.65, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.35), xytext=(0.35, 0.35), arrowprops=arrow_props)
    
    # Add conversion explanations
    ax.text(0.25, 0.55, '除2取余法\nDivide by 2', ha='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(0.75, 0.55, '除16取余法\nDivide by 16', ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(0.5, 0.2, '4位二进制=1位十六进制\n4 bits = 1 hex digit', ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('数制转换 Number System Conversion', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'number_conversion.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_bit_operations():
    """Create bit operations visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define the bit vectors
    a = '01101001'
    b = '01010101'
    
    operations = [
        ('AND (&)', [int(a[i]) & int(b[i]) for i in range(8)], '#FFB6C1'),
        ('OR (|)', [int(a[i]) | int(b[i]) for i in range(8)], '#98FB98'),
        ('XOR (^)', [int(a[i]) ^ int(b[i]) for i in range(8)], '#DDA0DD'),
        ('NOT (~)', [1 - int(b[i]) for i in range(8)], '#F0E68C')
    ]
    
    y_positions = [0.8, 0.6, 0.4, 0.2]
    
    # Draw input vectors
    ax.text(0.1, 0.95, 'A = ' + a, fontsize=14, fontweight='bold', fontfamily='monospace')
    ax.text(0.1, 0.9, 'B = ' + b, fontsize=14, fontweight='bold', fontfamily='monospace')
    
    for i, (op_name, result, color) in enumerate(operations):
        y = y_positions[i]
        
        # Operation label
        ax.text(0.05, y, op_name, fontsize=12, fontweight='bold', va='center')
        
        # Draw bit boxes
        for j in range(8):
            # Input A
            if i < 3:  # For operations involving A
                rect_a = patches.Rectangle((0.2 + j*0.08, y + 0.02), 0.06, 0.04, 
                                         linewidth=1, edgecolor='black', facecolor='lightblue')
                ax.add_patch(rect_a)
                ax.text(0.23 + j*0.08, y + 0.04, a[j], ha='center', va='center', 
                       fontfamily='monospace', fontweight='bold')
            
            # Input B (or just B for NOT)
            rect_b = patches.Rectangle((0.2 + j*0.08, y - 0.02), 0.06, 0.04, 
                                     linewidth=1, edgecolor='black', facecolor='lightgreen')
            ax.add_patch(rect_b)
            ax.text(0.23 + j*0.08, y, b[j], ha='center', va='center', 
                   fontfamily='monospace', fontweight='bold')
            
            # Result
            rect_r = patches.Rectangle((0.2 + j*0.08, y - 0.06), 0.06, 0.04, 
                                     linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect_r)
            ax.text(0.23 + j*0.08, y - 0.04, str(result[j]), ha='center', va='center', 
                   fontfamily='monospace', fontweight='bold')
        
        # Draw operation symbol
        if i < 3:
            ax.text(0.15, y, op_name[0], fontsize=16, fontweight='bold', ha='center', va='center')
        
        # Result value
        result_str = ''.join(map(str, result))
        ax.text(0.85, y - 0.02, f'= {result_str}', fontsize=12, fontweight='bold', 
               fontfamily='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1)
    ax.axis('off')
    ax.set_title('位运算可视化 Bitwise Operations Visualization', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bit_operations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_shift_operations():
    """Create shift operations diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Original values
    x1 = '01100010'  # 98 in decimal
    x2 = '10100010'  # -94 in 8-bit two's complement
    
    operations = [
        ('左移 << 3', ['00010000', '00010000'], '#FFE4B5'),
        ('逻辑右移 >> 2', ['00011000', '00101000'], '#E0FFFF'),
        ('算术右移 >> 2', ['00011000', '11101000'], '#F0E68C')
    ]
    
    # Header
    ax.text(0.1, 0.9, 'x₁ = ' + x1 + ' (正数)', fontsize=14, fontweight='bold', fontfamily='monospace')
    ax.text(0.6, 0.9, 'x₂ = ' + x2 + ' (负数)', fontsize=14, fontweight='bold', fontfamily='monospace')
    
    y_positions = [0.7, 0.5, 0.3]
    
    for i, (op_name, results, color) in enumerate(operations):
        y = y_positions[i]
        
        # Operation name
        ax.text(0.05, y, op_name, fontsize=12, fontweight='bold', va='center')
        
        # Draw bit boxes for both values
        for col, (x_val, result) in enumerate([(x1, results[0]), (x2, results[1])]):
            x_offset = 0.1 + col * 0.5
            
            # Original value
            for j in range(8):
                rect = patches.Rectangle((x_offset + j*0.04, y + 0.05), 0.035, 0.04, 
                                       linewidth=1, edgecolor='black', facecolor='lightgray')
                ax.add_patch(rect)
                ax.text(x_offset + j*0.04 + 0.0175, y + 0.07, x_val[j], 
                       ha='center', va='center', fontfamily='monospace', fontsize=10)
            
            # Arrow
            ax.annotate('', xy=(x_offset + 0.16, y - 0.02), xytext=(x_offset + 0.16, y + 0.02), 
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            
            # Result
            for j in range(8):
                rect = patches.Rectangle((x_offset + j*0.04, y - 0.05), 0.035, 0.04, 
                                       linewidth=2, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                ax.text(x_offset + j*0.04 + 0.0175, y - 0.03, result[j], 
                       ha='center', va='center', fontfamily='monospace', fontsize=10, fontweight='bold')
    
    # Add explanations
    ax.text(0.05, 0.1, '说明：\n• 左移：右边补0\n• 逻辑右移：左边补0\n• 算术右移：左边补符号位', 
           fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('移位运算 Shift Operations', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shift_operations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_signed_unsigned_conversion():
    """Create signed/unsigned conversion chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 4-bit examples
    bit_patterns = ['0000', '0001', '0111', '1000', '1111']
    signed_vals = [0, 1, 7, -8, -1]
    unsigned_vals = [0, 1, 7, 8, 15]
    
    # Create circular representation
    n_bits = 16
    angles = np.linspace(0, 2*np.pi, n_bits, endpoint=False)
    radius = 0.8
    
    # Draw circle
    circle = plt.Circle((0, 0), radius, fill=False, linewidth=2, color='black')
    ax.add_patch(circle)
    
    # Mark values around circle
    for i in range(n_bits):
        angle = angles[i]
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # 4-bit pattern
        bit_pattern = format(i, '04b')
        
        # Signed interpretation (4-bit two's complement)
        if i < 8:
            signed_val = i
        else:
            signed_val = i - 16
        
        # Unsigned interpretation
        unsigned_val = i
        
        # Color coding
        if i < 8:
            color = 'green'  # Positive
        else:
            color = 'red'    # Negative
        
        # Draw point
        ax.plot(x, y, 'o', color=color, markersize=8)
        
        # Add labels
        label_x = (radius + 0.15) * np.cos(angle)
        label_y = (radius + 0.15) * np.sin(angle)
        
        ax.text(label_x, label_y, f'{bit_pattern}\nS:{signed_val}\nU:{unsigned_val}', 
               ha='center', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add legend
    ax.plot([], [], 'o', color='green', markersize=10, label='正数区域 (0-7)')
    ax.plot([], [], 'o', color='red', markersize=10, label='负数区域 (8-15)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Add title and labels
    ax.set_title('4位有符号/无符号数转换\n4-bit Signed/Unsigned Conversion', 
                fontsize=14, fontweight='bold', pad=20)
    ax.text(0, -1.2, 'S: 有符号 (Signed)    U: 无符号 (Unsigned)', 
           ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signed_unsigned_conversion.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_byte_ordering():
    """Create byte ordering (endianness) diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Example value
    value = 0x01234567
    addresses = ['0x100', '0x101', '0x102', '0x103']
    big_endian = ['01', '23', '45', '67']
    little_endian = ['67', '45', '23', '01']
    
    # Big Endian
    y_big = 0.7
    ax.text(0.1, y_big + 0.1, '大端序 (Big Endian)', fontsize=14, fontweight='bold')
    ax.text(0.1, y_big + 0.05, '最高有效字节在最低地址', fontsize=10)
    
    for i, (addr, byte_val) in enumerate(zip(addresses, big_endian)):
        x = 0.2 + i * 0.15
        
        # Address
        ax.text(x + 0.05, y_big + 0.15, addr, ha='center', fontsize=10, fontweight='bold')
        
        # Memory cell
        rect = patches.Rectangle((x, y_big), 0.1, 0.08, 
                               linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        ax.text(x + 0.05, y_big + 0.04, byte_val, ha='center', va='center', 
               fontsize=12, fontweight='bold', fontfamily='monospace')
    
    # Arrow showing MSB to LSB
    ax.annotate('MSB → LSB', xy=(0.75, y_big + 0.04), xytext=(0.85, y_big + 0.04), 
               arrowprops=dict(arrowstyle='->', lw=2, color='red'), fontsize=10)
    
    # Little Endian
    y_little = 0.3
    ax.text(0.1, y_little + 0.1, '小端序 (Little Endian)', fontsize=14, fontweight='bold')
    ax.text(0.1, y_little + 0.05, '最低有效字节在最低地址', fontsize=10)
    
    for i, (addr, byte_val) in enumerate(zip(addresses, little_endian)):
        x = 0.2 + i * 0.15
        
        # Address
        ax.text(x + 0.05, y_little + 0.15, addr, ha='center', fontsize=10, fontweight='bold')
        
        # Memory cell
        rect = patches.Rectangle((x, y_little), 0.1, 0.08, 
                               linewidth=2, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(x + 0.05, y_little + 0.04, byte_val, ha='center', va='center', 
               fontsize=12, fontweight='bold', fontfamily='monospace')
    
    # Arrow showing LSB to MSB
    ax.annotate('LSB → MSB', xy=(0.75, y_little + 0.04), xytext=(0.85, y_little + 0.04), 
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'), fontsize=10)
    
    # Original value
    ax.text(0.5, 0.9, f'原始值: 0x{value:08X}', ha='center', fontsize=16, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7))
    
    # Usage examples
    ax.text(0.1, 0.1, '使用场景:\n大端序: Sun, PowerPC, 网络协议\n小端序: x86, ARM, Windows', 
           fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('字节序 (Byte Ordering / Endianness)', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'byte_ordering.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_integer_overflow():
    """Create integer overflow visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Unsigned overflow example (8-bit)
    max_val = 255
    overflow_example = 200 + 100  # Results in overflow
    
    # Create number line visualization
    x_vals = np.arange(0, 300, 10)
    y_unsigned = 0.7
    y_signed = 0.3
    
    # Unsigned overflow
    ax.text(0.1, y_unsigned + 0.15, '无符号数溢出 (8位)', fontsize=14, fontweight='bold')
    
    # Draw number line for unsigned
    ax.plot([0, 300], [y_unsigned, y_unsigned], 'k-', linewidth=2)
    
    # Mark overflow point
    ax.plot([255], [y_unsigned], 'ro', markersize=10, label='UMax = 255')
    ax.plot([300], [y_unsigned], 'bo', markersize=10, label='200 + 100 = 300')
    ax.plot([44], [y_unsigned], 'go', markersize=10, label='实际结果 = 44')
    
    # Draw wrap-around arrow
    ax.annotate('', xy=(44, y_unsigned + 0.05), xytext=(300, y_unsigned + 0.05), 
               arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                             connectionstyle="arc3,rad=0.3"))
    ax.text(172, y_unsigned + 0.08, '模运算回绕\n300 mod 256 = 44', ha='center', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Signed overflow (two's complement)
    ax.text(0.1, y_signed + 0.15, '有符号数溢出 (8位补码)', fontsize=14, fontweight='bold')
    
    # Draw number line for signed
    ax.plot([-150, 150], [y_signed, y_signed], 'k-', linewidth=2)
    
    # Mark overflow points
    ax.plot([-128], [y_signed], 'ro', markersize=10, label='TMin = -128')
    ax.plot([127], [y_signed], 'ro', markersize=10, label='TMax = 127')
    ax.plot([150], [y_signed], 'bo', markersize=10, label='100 + 50 = 150')
    ax.plot([-106], [y_signed], 'go', markersize=10, label='实际结果 = -106')
    
    # Draw wrap-around arrow
    ax.annotate('', xy=(-106, y_signed + 0.05), xytext=(150, y_signed + 0.05), 
               arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                             connectionstyle="arc3,rad=-0.3"))
    ax.text(22, y_signed + 0.08, '正溢出变负数\n150 - 256 = -106', ha='center', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Add scale labels
    for i in range(0, 301, 50):
        ax.text(i, y_unsigned - 0.03, str(i), ha='center', fontsize=8)
    
    for i in range(-150, 151, 50):
        ax.text(i, y_signed - 0.03, str(i), ha='center', fontsize=8)
    
    ax.set_xlim(-200, 350)
    ax.set_ylim(0.1, 1)
    ax.axis('off')
    ax.set_title('整数溢出行为 Integer Overflow Behavior', fontsize=16, fontweight='bold', pad=20)
    
    # Add explanation
    ax.text(0.5, 0.05, '关键概念：溢出时数值"回绕"到有效范围内', 
           ha='center', transform=ax.transAxes, fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'integer_overflow.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    print("创建ICS课程图片...")
    
    try:
        create_number_conversion()
        print("✓ 数制转换图创建完成")
        
        create_bit_operations()
        print("✓ 位运算图创建完成")
        
        create_shift_operations()
        print("✓ 移位操作图创建完成")
        
        create_signed_unsigned_conversion()
        print("✓ 有符号/无符号转换图创建完成")
        
        create_byte_ordering()
        print("✓ 字节序图创建完成")
        
        create_integer_overflow()
        print("✓ 整数溢出图创建完成")
        
        print(f"\n所有图片已保存到: {output_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已安装 matplotlib: pip install matplotlib")