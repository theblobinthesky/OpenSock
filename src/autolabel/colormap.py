from .timing import timed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_results(image, mapped_image_per_ch, hists_per_ch, bboxes, lut_per_ch, dm_per_ch, perc_change):
    fig, axes = plt.subplots(4, 6, figsize=(25, 5))

    for channel in range(3):
        mapped_image = mapped_image_per_ch[channel]
        hists = hists_per_ch[channel]
        lut = lut_per_ch[channel]
        dm = dm_per_ch[channel]

        axes[channel, 0].imshow(image[:, :, channel], cmap='gray')
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=plt.cm.tab10(idx), facecolor='none')
            axes[channel, 0].add_patch(rect)
        axes[channel, 0].set_title(f'Original Image (Channel {channel}, 10-bit)')
        axes[channel, 0].axis('off')
        
        axes[channel, 1].imshow(mapped_image, cmap='gray')
        axes[channel, 1].set_title('Mapped Image (8-bit)')
        axes[channel, 1].axis('off')
        
        colors = plt.cm.tab10(np.arange(len(bboxes)))
        for idx, hist in enumerate(hists[channel]):
            axes[channel, 2].plot(hist, color=colors[idx], label=f'BBox {idx}', alpha=0.7)
        axes[channel, 2].set_title(f'Original Spectra (Channel {channel}, 10-bit)')
        axes[channel, 2].set_xlabel('10-bit Value (0-1023)')
        axes[channel, 2].set_ylabel('Count')
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            cutout = mapped_image[y1:y2, x1:x2]
            hist, _ = np.histogram(cutout, bins=256, range=(0, 256))
            axes[channel, 3].plot(hist, color=colors[idx], label=f'BBox {idx}', alpha=0.7)
        axes[channel, 3].set_title(f'Modified Spectra (Channel {channel}, 8-bit)')
        axes[channel, 3].set_xlabel('8-bit Value (0-255)')
        axes[channel, 3].set_ylabel('Count')
        
        axes[channel, 4].plot(range(len(lut)), lut, 'b-')
        axes[channel, 4].set_title(f'Lookup Table (Channel {channel})')
        axes[channel, 4].set_xlabel('10-bit Input (0-1023)')
        axes[channel, 4].set_ylabel('8-bit Output (0-255)')
        axes[channel, 4].grid(True)

        axes[channel, 5].imshow(dm)
        axes[channel, 5].set_title('Dynamic Matrix')
        axes[channel, 5].axis('off')

    axes[3, 0].imshow(image / 1023.0)
    axes[3, 0].axis('off')

    axes[3, 1].imshow(np.stack(mapped_image_per_ch, axis=-1))
    axes[3, 1].axis('off')

    axes[3, 2].axis('off')
    axes[3, 2].axis('off')
    axes[3, 3].axis('off')
    axes[3, 4].axis('off')
    axes[3, 5].axis('off')

    fig.text(0.5, 0.05, f"relative to basic rounding we have {perc_change * 100:.2f}% more collisions", ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def generate_image_with_radial_fades(num_points, height=1080, width=1920, radius=400):
    img_accum = np.zeros((height, width, 3), dtype=np.float64)
    
    y, x = np.ogrid[:height, :width]
    
    centers = np.column_stack((np.random.randint(0, width, num_points),
                               np.random.randint(0, height, num_points)))
    colors = np.random.randint(0, 1024, (num_points, 3))
    
    for center, color in zip(centers, colors):
        cx, cy = center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        fade = np.maximum(0, 1 - dist / radius)
        for c in range(3):
            img_accum[:, :, c] += fade * color[c]
    
    img = np.clip(img_accum, 0, 1023).astype(np.uint64)
    for c in range(3):
        ch = img[:, :, c]
        ch[ch != 0] = 1023 - ch[ch != 0]
        ch[ch <= pixel_thresh] = 0

    return img

def generate_bboxes(num_bboxes, height=1080, width=1920, min_size=50, max_size=300):
    bboxes = []
    for _ in range(num_bboxes):
        w = np.random.randint(min_size, max_size + 1)
        h = np.random.randint(min_size, max_size + 1)
        x1 = np.random.randint(0, width - w + 1)
        y1 = np.random.randint(0, height - h + 1)
        bboxes.append((x1, y1, x1 + w, y1 + h))
    return bboxes

def extract_channel_hist(img, bbox, channel):
    x1, y1, x2, y2 = bbox
    cutout = img[y1:y2, x1:x2, channel]
    cutout = cutout[cutout >= pixel_thresh]
    hist = np.histogram(cutout, bins=1024, range=(0, 1024))[0]
    return hist

@timed
def calculate_optimal_lookup_table(hists: list, const_band_size: int):
    # The goal is to obtain a monotonic lookup table per color channel,
    # such that we preserve as much separation between hists as possible
    # when they are projected from 10-bit to 8-bit.
    # Solve this using dynamic programming, similar to the dynamic time warp algorithm.

    full_size, part_size = 1024, 256

    sum_hist = np.sum(np.stack(hists, axis=0), axis=0)
    cumsum_hist = np.cumsum(sum_hist, axis=0)
    max_query = np.zeros((full_size, full_size))
    for i in range(full_size):
        for j in range(full_size):
            if i <= j:
                max_query[i, j] = np.max(sum_hist[i:j + 1])


    dynamic_matrix = np.full((part_size, full_size), np.iinfo(np.uint64).max, dtype=np.uint64)
    sum_over_rest_matrix = np.full((part_size, full_size), np.iinfo(np.uint64).max, dtype=np.uint64)
    previous_matrix = np.zeros((part_size, full_size), np.uint16)

    # Run the dynamic programming algorithm.
    def get_min_idx(i, j):
        if i == 0 and j == 0:
            return 0, 1, 0

        a = dynamic_matrix[i - 1, j - 1]

        num_on_this_height = previous_matrix[i, j - 1]
        sum_over_rest = sum_over_rest_matrix[i, j - 1]
        sum_error = cumsum_hist[j] - cumsum_hist[j - num_on_this_height]
        max_error = max_query[j - num_on_this_height + 1, j]
        error_on_this_height = sum_error - max_error
        total_sum = error_on_this_height + sum_over_rest

        if (a < total_sum or j <= i) and y > 0:
            # In case j <= i you can only go diagonally: Enforce this case.

            # Notice how a unique i->j mapping does not introduce any error.
            # Only if i1, i2 with i1!=i2 and i1->j, i2->j exist does it cause 
            # disambiguation problems.
            return a + 0, 1, a
        else:
            return total_sum, 1 + num_on_this_height, sum_over_rest


    for y in range(part_size):
        for x in range(full_size):
            if y > x: continue
            y_linear_mapped = int(round(part_size * x / full_size))
            if abs(y - y_linear_mapped) > const_band_size: continue

            d, p, s = get_min_idx(y, x)
            dynamic_matrix[y, x] = d
            previous_matrix[y, x] = p
            sum_over_rest_matrix[y, x] = s


    # Backtrack to find the optimal monotonically increasing lookup table.
    i, j = part_size - 1, full_size - 1
    lookup_table = []
    while i >= 0 and j >= 0:
        lookup_table.append((i, j))
        num_on_this_height = previous_matrix[i, j]

        if num_on_this_height == 1:
            i, j = i - 1, j - 1
        else:
            j = j - 1
    
    lookup_table = [to for (to, _) in lookup_table]
    lookup_table.reverse()

    return dynamic_matrix, np.array(lookup_table)


@timed
def calculate_luts(image: np.ndarray, masks: list[np.ndarray], const_band_size: int=64, full_bins: int=1024, part_bins: int=256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert part_bins == 256
    luts = []

    for c in range(3):

        hists = []
        for mask in masks:
            channel = image[:, :, c]
            hist = np.histogram(channel[mask], bins=full_bins, range=(0, full_bins))[0]
            hists.append(hist)

        _, lut = calculate_optimal_lookup_table(hists, const_band_size=const_band_size)
        luts.append(lut)

    return tuple(luts)

def apply_luts(image: np.ndarray, luts: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mapped = []
    for c in range(3):
        channel = image[:, :, c]
        lut = luts[c]
        mapped.append(lut[channel])

    return np.stack(mapped, axis=-1)


if __name__ == "__main__":
    np.random.seed(4)
    num_points = 10
    num_bboxes = 10
    const_band_size = 100
    pixel_thresh = 200

    image = generate_image_with_radial_fades(num_points)
    bboxes = generate_bboxes(num_bboxes)

    mapped_image_per_ch = []
    hists_per_ch = []
    hists_per_ch = []
    lut_per_ch = []
    dm_per_ch = []
    avg_perc_change = 0.0

    for channel in range(3):
        hists = [[extract_channel_hist(image, bbox, channel) for bbox in bboxes] for channel in range(3)]
        dynamic_matrix, lut = calculate_optimal_lookup_table(hists[channel], const_band_size=const_band_size)

        mapped_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mapped_image = lut[image[:, :, channel]]

        dynamic_matrix[dynamic_matrix == np.iinfo(np.uint64).max] = 0
        sum_hist = np.sum(np.concat(hists, axis=0), axis=0)

        def get_avg_collisions(lut):
            avg_collisions = 0.0
            for full_idx in range(256):
                indices = np.where(lut == full_idx)[0]
                if indices.size:
                    hist_vals = sum_hist[indices]
                    avg_collisions += (np.sum(hist_vals) - np.max(hist_vals)) / 1024.0
            return avg_collisions

        basic_lut = np.round(np.linspace(0, 255, num=1024)).astype(np.uint32)
        basic_cols = get_avg_collisions(basic_lut)
        lut_cols = get_avg_collisions(lut)
        avg_perc_change += (lut_cols - basic_cols) / basic_cols

        mapped_image_per_ch.append(mapped_image)
        hists_per_ch.append(hists)
        lut_per_ch.append(lut)
        dm_per_ch.append(dynamic_matrix)

    avg_perc_change /= 3

    visualize_results(image, mapped_image_per_ch, hists_per_ch, bboxes, lut_per_ch, dm_per_ch, avg_perc_change)
