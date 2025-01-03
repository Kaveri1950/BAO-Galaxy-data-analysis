import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
dataset1 = np.loadtxt('Galaxy Sample 1.txt')  # Load Dataset 1
dataset2 = np.loadtxt('Galaxy Sample 2.txt')  # Load Dataset 2

# Process Dataset 1
N_rows1 = dataset1.shape[0]  # Number of rows (galaxies) in Dataset 1
N_cols1 = dataset1.shape[1]  # Number of columns (coordinates) in Dataset 1

# Iterate through Dataset 1
for galaxy_indx in range(N_rows1):
    x1 = dataset1[galaxy_indx, 0]  # x-coordinate
    y1 = dataset1[galaxy_indx, 1]  # y-coordinate
    z1 = dataset1[galaxy_indx, 2]  # z-coordinate

    # Print for the first 5 galaxies
    if galaxy_indx < 5:
        print(f"Dataset 1 - Galaxy {galaxy_indx + 1}: x = {x1:.2f}, y = {y1:.2f}, z = {z1:.2f}")

print("\n")  # Separator for clarity

# Process Dataset 2
N_rows2 = dataset2.shape[0]  # Number of rows (galaxies) in Dataset 2
N_cols2 = dataset2.shape[1]  # Number of columns (coordinates) in Dataset 2

# Iterate through Dataset 2
for galaxy_indx in range(N_rows2):
    x2 = dataset2[galaxy_indx, 0]  # x-coordinate
    y2 = dataset2[galaxy_indx, 1]  # y-coordinate
    z2 = dataset2[galaxy_indx, 2]  # z-coordinate

    # Print for the first 5 galaxies
    if galaxy_indx < 5:
        print(f"Dataset 2 - Galaxy {galaxy_indx + 1}: x = {x2:.2f}, y = {y2:.2f}, z = {z2:.2f}")

# Function to compute pairwise distances
def compute_pairwise_distances(dataset):
    N = dataset.shape[0]  # Number of galaxies
    distances = []  # Initialize an empty list to store distances

    # Nested loops to compute the distance between all pairs
    for i in range(N):  # Outer loop for the first galaxy
        for j in range(i + 1, N):  # Inner loop for the second galaxy (j > i to avoid repetition)
            x1, y1, z1 = dataset[i]  # Coordinates of the first galaxy
            x2, y2, z2 = dataset[j]  # Coordinates of the second galaxy

            # Compute Euclidean distance
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


            # Append the distance to the list
            distances.append(distance)

    return np.array(distances)  # Convert the list to a NumPy array for further processing

# Compute pairwise distances for both datasets
dist_pair_arr_1 = compute_pairwise_distances(dataset1)  # Pairwise distances for Dataset 1
dist_pair_arr_2 = compute_pairwise_distances(dataset2)  # Pairwise distances for Dataset 2

N1 = dataset1.shape[0]  # Number of galaxies in Dataset 1
N2 = dataset2.shape[0]  # Number of galaxies in Dataset 2

# Calculate the expected length of dist_pair_arr_1 (pairs of galaxies from Dataset 1)
expected_length1 = N1 * (N1 - 1) // 2  # This calculates the number of pairs in Dataset 1

# Calculate the expected length of dist_pair_arr_2 (pairs of galaxies from Dataset 2)
expected_length2 = N2 * (N2 - 1) // 2  # This calculates the number of pairs in Dataset 2

# Printing the length of dist_pair_arr_1 and dist_pair_arr_2
print(f"length of dist_pair_arr_1 for Dataset 1 = {len(dist_pair_arr_1)}")
print(f"Expected length of dist_pair_arr_1 = {expected_length1}")
print(f"length of dist_pair_arr_2 for Dataset 2 = {len(dist_pair_arr_2)}")
print(f"Expected length of dist_pair_arr_2 = {expected_length2}")

# Function to plot histograms
def plot_histogram(distances, title):
    # Define parameters for the histogram
    bins = 50  # Number of bins
    range_max = 200  # Maximum range for histogram

    # Compute the histogram
    hist, bin_edges = np.histogram(distances, bins=bins, range=(0, range_max))

    # Normalize the histogram
    norm_hist = hist / np.sum(hist)

    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, norm_hist, label='Normalized Histogram', marker='o')
    plt.xlabel('Distance r (Mpc)', fontsize=12)
    plt.ylabel('Frequency (Normalized)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid()
    plt.show()

    return bin_centers, norm_hist

# Plot histograms for both datasets
bin_centers1, norm_hist1 = plot_histogram(dist_pair_arr_1, "Dataset 1: Pairwise Distances")
bin_centers2, norm_hist2 = plot_histogram(dist_pair_arr_2, "Dataset 2: Pairwise Distances")

# Function to plot histograms together
def plot_histogram(distances_1, distances_2, title_1, title_2):
    # Define parameters for the histogram
    bins = 50  # Number of bins
    range_max = 200  # Maximum range for histogram

    # Compute the histograms for both datasets
    hist1, bin_edges1 = np.histogram(distances_1, bins=bins, range=(0, range_max))
    hist2, bin_edges2 = np.histogram(distances_2, bins=bins, range=(0, range_max))

    # Normalize the histograms
    norm_hist1 = hist1 / np.sum(hist1)
    norm_hist2 = hist2 / np.sum(hist2)

    # Calculate bin centers
    bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])

    # Plot the histograms on the same axes
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers1, norm_hist1, label=title_1, marker='o', linestyle='-', color='blue')
    plt.plot(bin_centers2, norm_hist2, label=title_2, marker='o', linestyle='--', color='red')

    # Customize the plot
    plt.xlabel('Distance r (Mpc)', fontsize=12)
    plt.ylabel('Frequency (Normalized)', fontsize=12)
    plt.title('Comparison of Pairwise Distances', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

    return bin_centers1, norm_hist1, bin_centers2, norm_hist2

# Example usage
bin_centers1, norm_hist1, bin_centers2, norm_hist2 = plot_histogram(dist_pair_arr_1, dist_pair_arr_2, "Dataset 1: Pairwise Distances", "Dataset 2: Pairwise Distances")

# Function to find the BAO peak
def find_bao_peak(bin_centers, norm_hist):
    # Create a boolean array where True indicates a local maximum
    local_maxima = (norm_hist[1:-1] > norm_hist[:-2]) & (norm_hist[1:-1] > norm_hist[2:])

    # Find indices of the local maxima (excluding the boundary points)
    peak_indices = np.where(local_maxima)[0] + 1  # +1 to account for the exclusion of boundaries

    if len(peak_indices) > 0:
        # Find the index of the peak with the maximum value
        peak_location = bin_centers[peak_indices[np.argmax(norm_hist[peak_indices])]]
    else:
        # If no peak is found, return None
        peak_location = None

    return peak_location

# Find BAO peaks for both datasets
peak1 = find_bao_peak(bin_centers1, norm_hist1)
peak2 = find_bao_peak(bin_centers2, norm_hist2)

# Print the results, handling the case where no peak is found
print(f"BAO Peak for Dataset 1 occurs at r = {peak1:.2f} Mpc" if peak1 is not None else "No peak found for Dataset 1")
print(f"BAO Peak for Dataset 2 occurs at r = {peak2:.2f} Mpc" if peak2 is not None else "No peak found for Dataset 2")

# Comparison of BAO peaks and histogram smoothness
print(f"Comparison of BAO peaks:")
print(f"Dataset 1 peak: {peak1:.2f} Mpc")
print(f"Dataset 2 peak: {peak2:.2f} Mpc")
print(f"Difference in smoothness of histograms:")
print(f"Dataset 1 has fewer galaxies (N=1000), resulting in more noise.")
print(f"Dataset 2 has more galaxies (N=5000), resulting in a smoother curve.")
