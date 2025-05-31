import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import random
import colorsys
import math

# Function to generate unique distinguishable colors for threads


def generate_unique_colors(n):
    colors = []
    for i in range(n):
        # Use HSV color space to generate distinguishable colors
        hue = i / n
        saturation = 0.8 + 0.2 * (i % 2)  # Alternate between 0.8 and 1.0
        value = 0.7 + 0.3 * ((i // 2) % 2)  # Alternate between 0.7 and 1.0

        # Convert to RGB and then to hex
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#' + ''.join([f'{int(255*x):02x}' for x in rgb])
        colors.append(hex_color)
    return colors


def rgb_matrix_from_hex(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return np.array([[tuple(int(matrix[i][j][k:k+2], 16) for k in (1, 3, 5))
                     for j in range(cols)] for i in range(rows)], dtype=np.uint8)

# Main simulation function


def main():
    # Simulation parameters
    global BLOCK_SIZE
    BLOCK_SIZE = 4
    size = 16  # Smaller size for better visualization

    # Set up the template parameters
    RESULTS_PER_THREAD_X = 1
    RESULTS_PER_THREAD_Y = 2

    # Calculate grid dimensions based on template parameters
    threads_per_block_x = BLOCK_SIZE
    threads_per_block_y = BLOCK_SIZE

    num_blocks_x = (size + BLOCK_SIZE * RESULTS_PER_THREAD_X -
                    1) // (BLOCK_SIZE * RESULTS_PER_THREAD_X)
    num_blocks_y = (size + BLOCK_SIZE * RESULTS_PER_THREAD_Y -
                    1) // (BLOCK_SIZE * RESULTS_PER_THREAD_Y)

    # Print information about the simulation
    print(f"Matrix size: {size}x{size}")
    print(f"Block size: {BLOCK_SIZE}x{BLOCK_SIZE}")
    print(f"Results per thread: {RESULTS_PER_THREAD_X}x{RESULTS_PER_THREAD_Y}")
    print(f"Grid dimensions: {num_blocks_x}x{num_blocks_y} blocks")
    print(
        f"Block dimensions: {threads_per_block_x}x{threads_per_block_y} threads")

    # Calculate total number of threads for color generation
    total_blocks = num_blocks_x * num_blocks_y
    total_threads_per_block = threads_per_block_x * threads_per_block_y
    total_threads = total_blocks * total_threads_per_block

    # Generate unique colors for each thread
    thread_colors = generate_unique_colors(total_threads)

    # Initialize matrices
    A = [['#FFFFFF' for _ in range(size)] for _ in range(size)]
    B = [['#FFFFFF' for _ in range(size)] for _ in range(size)]
    C = [['#000000' for _ in range(size)] for _ in range(size)]

    # For storing compute patterns
    compute_patterns = []

    # Track block computations for visualization
    thread_blocks = np.zeros((size, size), dtype=int)

    def simulate_matrix_mul(blockIdx_y, blockIdx_x):
        # Shared memory for the block
        As = [['#FFFFFF' for _ in range(BLOCK_SIZE)] for _ in range(
            BLOCK_SIZE * RESULTS_PER_THREAD_Y)]
        Bs = [['#FFFFFF' for _ in range(
            BLOCK_SIZE * RESULTS_PER_THREAD_X)] for _ in range(BLOCK_SIZE)]

        # For each thread in the block
        for ty in range(threads_per_block_y):
            for tx in range(threads_per_block_x):
                baseRow = (blockIdx_y * threads_per_block_y + ty) * \
                    RESULTS_PER_THREAD_Y
                baseCol = (blockIdx_x * threads_per_block_x + tx) * \
                    RESULTS_PER_THREAD_X

                # Calculate thread ID and color
                block_id = blockIdx_y * num_blocks_x + blockIdx_x
                thread_id_in_block = ty * threads_per_block_x + tx
                global_thread_id = block_id * total_threads_per_block + thread_id_in_block
                thread_color = thread_colors[global_thread_id % len(
                    thread_colors)]

                # Initialize result accumulators
                sum_values = [[0.0 for _ in range(RESULTS_PER_THREAD_X)] for _ in range(
                    RESULTS_PER_THREAD_Y)]

                # Process tiles
                numTiles = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
                for m in range(numTiles):
                    # Load A tile to shared memory
                    for y in range(RESULTS_PER_THREAD_Y):
                        row = baseRow + y
                        if row < size and (m * BLOCK_SIZE + tx) < size:
                            As[ty * RESULTS_PER_THREAD_Y + y][tx] = thread_color
                            A[row][m * BLOCK_SIZE + tx] = thread_color

                    # Load B tile to shared memory
                    for x in range(RESULTS_PER_THREAD_X):
                        col = baseCol + x
                        if (m * BLOCK_SIZE + ty) < size and col < size:
                            Bs[ty][tx * RESULTS_PER_THREAD_X + x] = thread_color
                            B[m * BLOCK_SIZE + ty][col] = thread_color

                    # Compute partial dot products
                    for k in range(BLOCK_SIZE):
                        for y in range(RESULTS_PER_THREAD_Y):
                            row = baseRow + y
                            if row >= size:
                                continue

                            for x in range(RESULTS_PER_THREAD_X):
                                col = baseCol + x
                                if col >= size:
                                    continue

                                # Record computation pattern for first few blocks
                                if m == 0 and blockIdx_y < 2 and blockIdx_x < 2:
                                    compute_patterns.append({
                                        'thread': (blockIdx_y, blockIdx_x, ty, tx),
                                        'result_cell': (row, col),
                                        'a_val_idx': (row, m * BLOCK_SIZE + k),
                                        'b_val_idx': (m * BLOCK_SIZE + k, col),
                                        'thread_color': thread_color
                                    })

                                # Simulate dot product accumulation
                                sum_values[y][x] += 1

                # Write results
                for y in range(RESULTS_PER_THREAD_Y):
                    row = baseRow + y
                    if row < size:
                        for x in range(RESULTS_PER_THREAD_X):
                            col = baseCol + x
                            if col < size:
                                C[row][col] = thread_color
                                thread_blocks[row, col] = block_id

        return As, Bs

    # Store shared memory blocks for visualization
    all_As_blocks = []
    all_Bs_blocks = []
    block_titles_As = []
    block_titles_Bs = []

    # Run simulation for all blocks
    for blockIdx_y in range(num_blocks_y):
        for blockIdx_x in range(num_blocks_x):
            print(f"Simulating block ({blockIdx_y}, {blockIdx_x})...")
            As, Bs = simulate_matrix_mul(blockIdx_y, blockIdx_x)

            all_As_blocks.append(As)
            all_Bs_blocks.append(Bs)
            block_titles_As.append(f"As ({blockIdx_y},{blockIdx_x})")
            block_titles_Bs.append(f"Bs ({blockIdx_y},{blockIdx_x})")

    # Create annotation for C matrix showing thread IDs
    C_annotations = {}
    for y in range(size):
        for x in range(size):
            block_y = y // (BLOCK_SIZE * RESULTS_PER_THREAD_Y)
            block_x = x // (BLOCK_SIZE * RESULTS_PER_THREAD_X)
            thread_y = (y // RESULTS_PER_THREAD_Y) % BLOCK_SIZE
            thread_x = (x // RESULTS_PER_THREAD_X) % BLOCK_SIZE
            C_annotations[(y, x)] = f"({thread_y},{thread_x})"

    print("Generating visualizations in 3 windows...")

    # WINDOW 1: Main matrices A, B, and C
    fig1 = plt.figure(figsize=(18, 6))
    fig1.suptitle(
        "CUDA Matrix Multiplication - Input and Output Matrices", fontsize=16)

    # Matrix A
    ax_A = fig1.add_subplot(1, 3, 1)
    ax_A.set_title(
        "Matrix A - Elements colored by accessing threads", fontsize=12)

    # Add grid lines for better block visualization
    for i in range(0, size, BLOCK_SIZE):
        ax_A.axhline(y=i-0.5, color='black', linestyle='-', linewidth=1.5)
    for j in range(0, size, BLOCK_SIZE):
        ax_A.axvline(x=j-0.5, color='black', linestyle='-', linewidth=1.5)

    rgb_A = rgb_matrix_from_hex(A)
    ax_A.imshow(rgb_A)
    ax_A.set_xticks(np.arange(size))
    ax_A.set_yticks(np.arange(size))
    ax_A.set_xticklabels(np.arange(size))
    ax_A.set_yticklabels(np.arange(size))
    ax_A.tick_params(axis='both', which='both', length=0)

    # Matrix B
    ax_B = fig1.add_subplot(1, 3, 2)
    ax_B.set_title(
        "Matrix B - Elements colored by accessing threads", fontsize=12)

    # Add grid lines
    for i in range(0, size, BLOCK_SIZE):
        ax_B.axhline(y=i-0.5, color='black', linestyle='-', linewidth=1.5)
    for j in range(0, size, BLOCK_SIZE):
        ax_B.axvline(x=j-0.5, color='black', linestyle='-', linewidth=1.5)

    rgb_B = rgb_matrix_from_hex(B)
    ax_B.imshow(rgb_B)
    ax_B.set_xticks(np.arange(size))
    ax_B.set_yticks(np.arange(size))
    ax_B.set_xticklabels(np.arange(size))
    ax_B.set_yticklabels(np.arange(size))
    ax_B.tick_params(axis='both', which='both', length=0)

    # Matrix C
    ax_C = fig1.add_subplot(1, 3, 3)
    ax_C.set_title(
        "Matrix C - Elements colored by computing threads", fontsize=12)

    # Add grid lines
    for i in range(0, size, BLOCK_SIZE):
        ax_C.axhline(y=i-0.5, color='black', linestyle='-', linewidth=1.5)
    for j in range(0, size, BLOCK_SIZE):
        ax_C.axvline(x=j-0.5, color='black', linestyle='-', linewidth=1.5)

    rgb_C = rgb_matrix_from_hex(C)
    ax_C.imshow(rgb_C)
    ax_C.set_xticks(np.arange(size))
    ax_C.set_yticks(np.arange(size))
    ax_C.set_xticklabels(np.arange(size))
    ax_C.set_yticklabels(np.arange(size))
    ax_C.tick_params(axis='both', which='both', length=0)

    # Add annotations for thread IDs if matrix is small enough
    if size <= 16:
        for i in range(size):
            for j in range(size):
                if (i, j) in C_annotations:
                    text = C_annotations[(i, j)]
                    # Calculate text color for contrast
                    r, g, b = [int(C[i][j][k:k+2], 16)/255 for k in (1, 3, 5)]
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b
                    text_color = 'black' if brightness > 0.5 else 'white'
                    ax_C.text(j, i, text, ha='center', va='center',
                              color=text_color, fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # WINDOW 2: Shared memory blocks and thread block responsibilities
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle(
        "CUDA Matrix Multiplication - Memory Access and Thread Block Distribution", fontsize=16)

    # Thread block responsibilities
    ax_blocks = fig2.add_subplot(2, 1, 1)
    ax_blocks.set_title(
        'Thread Block Responsibilities for Matrix C', fontsize=14)

    # Display the block ID matrix with a discrete colormap
    cmap = plt.cm.get_cmap('tab20', max(thread_blocks.max() + 1, 20))
    im = ax_blocks.imshow(thread_blocks, cmap=cmap)

    # Add grid lines for blocks
    for i in range(0, size, BLOCK_SIZE * RESULTS_PER_THREAD_Y):
        ax_blocks.axhline(y=i-0.5, color='black', linestyle='-', linewidth=2.0)
    for j in range(0, size, BLOCK_SIZE * RESULTS_PER_THREAD_X):
        ax_blocks.axvline(x=j-0.5, color='black', linestyle='-', linewidth=2.0)

    # Add grid lines for individual threads
    for i in range(0, size, RESULTS_PER_THREAD_Y):
        # Skip if already drawn as block boundary
        if i % (BLOCK_SIZE * RESULTS_PER_THREAD_Y) != 0:
            ax_blocks.axhline(y=i-0.5, color='gray',
                              linestyle='-', linewidth=0.5)
    for j in range(0, size, RESULTS_PER_THREAD_X):
        # Skip if already drawn as block boundary
        if j % (BLOCK_SIZE * RESULTS_PER_THREAD_X) != 0:
            ax_blocks.axvline(x=j-0.5, color='gray',
                              linestyle='-', linewidth=0.5)

    # Add annotations for thread IDs (for smaller matrices only)
    if size <= 16:
        for i in range(size):
            for j in range(size):
                ax_blocks.text(j, i, C_annotations[(i, j)], ha='center', va='center',
                               color='white', fontsize=8)

    # Add colorbar for block IDs
    cbar = plt.colorbar(im, ax=ax_blocks)
    cbar.set_label('Block ID')

    # Add indices
    ax_blocks.set_xticks(np.arange(size))
    ax_blocks.set_yticks(np.arange(size))
    ax_blocks.set_xticklabels(np.arange(size))
    ax_blocks.set_yticklabels(np.arange(size))
    ax_blocks.tick_params(axis='both', which='both', length=0)

    # Shared memory blocks (in a grid layout)
    # Show at most 8 blocks to avoid overcrowding
    num_blocks_to_show = min(8, len(all_As_blocks))

    # Create a grid for shared memory blocks
    grid = GridSpec(2, num_blocks_to_show, bottom=0.05,
                    top=0.45, hspace=0.3, wspace=0.2)

    # Shared memory As blocks
    for i in range(min(num_blocks_to_show, len(all_As_blocks))):
        ax = fig2.add_subplot(grid[0, i])
        ax.set_title(block_titles_As[i], fontsize=10)

        block_height = BLOCK_SIZE * RESULTS_PER_THREAD_Y
        block_width = BLOCK_SIZE

        rgb_block = rgb_matrix_from_hex(all_As_blocks[i])
        ax.imshow(rgb_block)

        # Add grid lines
        for row in range(0, block_height, RESULTS_PER_THREAD_Y):
            ax.axhline(y=row-0.5, color='black', linestyle='-', linewidth=0.5)

        # Keep ticks only for smaller blocks
        if block_height <= 8 and block_width <= 8:
            ax.set_xticks(np.arange(block_width))
            ax.set_yticks(np.arange(block_height))
            ax.set_xticklabels(np.arange(block_width))
            ax.set_yticklabels(np.arange(block_height))
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if i == 0:
            ax.set_ylabel("Shared Memory As", fontsize=12)

    # Shared memory Bs blocks
    for i in range(min(num_blocks_to_show, len(all_Bs_blocks))):
        ax = fig2.add_subplot(grid[1, i])
        ax.set_title(block_titles_Bs[i], fontsize=10)

        block_height = BLOCK_SIZE
        block_width = BLOCK_SIZE * RESULTS_PER_THREAD_X

        rgb_block = rgb_matrix_from_hex(all_Bs_blocks[i])
        ax.imshow(rgb_block)

        # Add grid lines
        for col in range(0, block_width, RESULTS_PER_THREAD_X):
            ax.axvline(x=col-0.5, color='black', linestyle='-', linewidth=0.5)

        # Keep ticks only for smaller blocks
        if block_height <= 8 and block_width <= 8:
            ax.set_xticks(np.arange(block_width))
            ax.set_yticks(np.arange(block_height))
            ax.set_xticklabels(np.arange(block_width))
            ax.set_yticklabels(np.arange(block_height))
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if i == 0:
            ax.set_ylabel("Shared Memory Bs", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # WINDOW 3: Computation patterns visualization
    # Extract unique result cells and their patterns
    result_cells = {}
    for pattern in compute_patterns:
        cell = pattern['result_cell']
        if cell not in result_cells:
            result_cells[cell] = []
        result_cells[cell].append(pattern)

    # Create plots for some sample cells to show computation patterns
    sample_cells = list(result_cells.keys())[:min(4, len(result_cells))]

    if sample_cells:
        fig3 = plt.figure(figsize=(16, 10))
        fig3.suptitle(
            'CUDA Matrix Multiplication - Computation Patterns for Sample C Matrix Elements', fontsize=16)

        for idx, cell in enumerate(sample_cells):
            patterns = result_cells[cell]
            row, col = cell

            # Create A and B matrices with data access patterns for this result
            A_access = [['#FFFFFF' for _ in range(size)] for _ in range(size)]
            B_access = [['#FFFFFF' for _ in range(size)] for _ in range(size)]

            for pattern in patterns:
                a_row, a_col = pattern['a_val_idx']
                b_row, b_col = pattern['b_val_idx']
                thread_color = pattern['thread_color']

                A_access[a_row][a_col] = thread_color
                B_access[b_row][b_col] = thread_color

            # Create subplot for A
            ax_a = fig3.add_subplot(2, len(sample_cells), idx + 1)
            rgb_a = rgb_matrix_from_hex(A_access)
            ax_a.imshow(rgb_a)
            ax_a.set_title(f'A elements used for C[{row},{col}]', fontsize=12)

            # Add grid lines
            for i in range(0, size, BLOCK_SIZE):
                ax_a.axhline(y=i-0.5, color='black',
                             linestyle='-', linewidth=0.8)
            for j in range(0, size, BLOCK_SIZE):
                ax_a.axvline(x=j-0.5, color='black',
                             linestyle='-', linewidth=0.8)

            # Add highlight for the specific row
            ax_a.axhline(y=row, color='red', linestyle='-', linewidth=0.5)

            # Create subplot for B
            ax_b = fig3.add_subplot(
                2, len(sample_cells), idx + 1 + len(sample_cells))
            rgb_b = rgb_matrix_from_hex(B_access)
            ax_b.imshow(rgb_b)
            ax_b.set_title(f'B elements used for C[{row},{col}]', fontsize=12)

            # Add grid lines
            for i in range(0, size, BLOCK_SIZE):
                ax_b.axhline(y=i-0.5, color='black',
                             linestyle='-', linewidth=0.8)
            for j in range(0, size, BLOCK_SIZE):
                ax_b.axvline(x=j-0.5, color='black',
                             linestyle='-', linewidth=0.8)

            # Add highlight for the specific column
            ax_b.axvline(x=col, color='red', linestyle='-', linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show all figures
    plt.show()


if __name__ == "__main__":
    main()
