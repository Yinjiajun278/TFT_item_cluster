import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os
from typing import List, Dict, Set, Tuple
import heapq

def load_normal_items(file_path: str) -> Set[str]:
    with open(file_path, 'r') as f:
        return set(item.strip() for item in f.readlines())

def load_legal_items(file_path: str) -> Set[str]:
    with open(file_path, 'r') as f:
        return set(item.strip() for item in f.readlines())

def extract_builds(file_content: str, legal_items: Set[str], normal_items: Set[str]) -> List[List[str]]:
    builds = []
    current_build = []
    lines = file_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line in legal_items and line in normal_items:
            current_build.append(line)
        else:
            if len(current_build) == 3:
                builds.append(current_build)
            current_build = []
    
    if len(current_build) == 3:
        builds.append(current_build)
    
    return builds

def process_character_file(file_path: str, legal_items: Set[str], normal_items: Set[str]) -> List[List[str]]:
    with open(file_path, 'r') as f:
        content = f.read()
    return extract_builds(content, legal_items, normal_items)

def process_directory(directory: str, legal_items_file: str, normal_items_file: str) -> Dict[str, List[List[str]]]:
    legal_items = load_legal_items(legal_items_file)
    normal_items = load_normal_items(normal_items_file)
    character_builds = {}

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            character_name = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            builds = process_character_file(file_path, legal_items, normal_items)
            if builds:
                character_builds[character_name] = builds

    return character_builds

def find_duplicate_items(builds: List[List[str]]) -> Dict[str, int]:
    item_max_count = defaultdict(int)
    for build in builds:
        build_counts = defaultdict(int)
        for item in build:
            build_counts[item] += 1
            item_max_count[item] = max(item_max_count[item], build_counts[item])
    
    return {item: count for item, count in item_max_count.items() if count > 1}

def create_co_occurrence_matrix(character_builds: Dict[str, List[List[str]]], all_items: List[str]) -> Tuple[np.ndarray, List[str]]:
    item_index = {item: i for i, item in enumerate(all_items)}
    matrix = np.zeros((len(all_items), len(all_items)))
    
    for character, builds in character_builds.items():
        for build in builds:
            for item1 in build:
                for item2 in build:
                    if item1 != item2:
                        i, j = item_index[item1], item_index[item2]
                        matrix[i, j] += 1
                        matrix[j, i] += 1
    
    return matrix, all_items

def hierarchical_clustering(similarity_matrix: np.ndarray) -> np.ndarray:
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='average')
    
    return linkage_matrix

def get_clusters_at_all_levels(linkage_matrix: np.ndarray, n_items: int) -> List[List[int]]:
    all_clusters = []
    for t in range(n_items):
        clusters = fcluster(linkage_matrix, t=t, criterion='distance')
        all_clusters.append(clusters)
    return all_clusters

def identify_core_items(character_builds: List[List[str]], threshold: float = 0.7) -> Set[str]:
    item_counts = defaultdict(int)
    total_builds = len(character_builds)
    
    for build in character_builds:
        for item in build:
            item_counts[item] += 1
    
    core_items = {item for item, count in item_counts.items() 
                  if count / total_builds >= threshold}
    
    return core_items

def find_highest_level_cluster(item_index: int, all_clusters: List[List[int]], 
                               character_items: Set[str], all_items: List[str], 
                               core_items: Set[str], threshold: float = 0.75) -> List[str]:
    for clusters in reversed(all_clusters):
        cluster = [i for i, c in enumerate(clusters) if c == clusters[item_index]]
        cluster_items = [all_items[i] for i in cluster]
        
        # Limit cluster size to 3 items
        if len(cluster_items) > 3:
            continue
        
        non_core_cluster_items = [item for item in cluster_items if item not in core_items]
        non_core_character_items = character_items - core_items
        
        if len(non_core_cluster_items) > 0:
            if len(set(non_core_cluster_items) & non_core_character_items) / len(non_core_cluster_items) >= threshold:
                return cluster_items
        else:
            # If all items in the cluster are core items, check if they're all used by the character
            if set(cluster_items).issubset(character_items):
                return cluster_items
    
    return [all_items[item_index]]

def get_cluster_hierarchy(linkage_matrix: np.ndarray, all_items: List[str]) -> List[Tuple[int, List[str]]]:
    n = len(all_items)
    clusters = [{i} for i in range(n)]
    hierarchy = []

    for i, (c1, c2, _, _) in enumerate(linkage_matrix):
        c1, c2 = int(c1), int(c2)
        new_cluster = clusters[c1] | clusters[c2]
        clusters.append(new_cluster)
        
        hierarchy.append((n + i, [all_items[i] for i in new_cluster]))

    return sorted(hierarchy, key=lambda x: len(x[1]), reverse=True)

def analyze_builds(character_builds: Dict[str, List[List[str]]]) -> Tuple[Dict[str, Tuple[List[str], Set[str], Set[Tuple[str, ...]], Dict[str, int]]], List[Tuple[int, List[str]]]]:
    all_items = list(set(item for builds in character_builds.values() for build in builds for item in build))
    
    results = {}
    for character, builds in character_builds.items():
        character_items = set(item for build in builds for item in build)
        core_items = identify_core_items(builds)
        duplicate_items = find_duplicate_items(builds)
        
        # Remove core items from builds for cluster analysis
        non_core_builds = [[item for item in build if item not in core_items] for build in builds]
        non_core_items = list(set(item for build in non_core_builds for item in build))
        
        if non_core_items:  # Only perform cluster analysis if there are non-core items
            co_occurrence_matrix, items = create_co_occurrence_matrix({character: non_core_builds}, non_core_items)
            similarity_matrix = cosine_similarity(co_occurrence_matrix)
            linkage_matrix = hierarchical_clustering(similarity_matrix)
            all_clusters = get_clusters_at_all_levels(linkage_matrix, len(non_core_items))
            
            character_clusters = set()
            item_counts = defaultdict(int)
            for build in non_core_builds:
                for item in build:
                    item_counts[item] += 1
            
            # Sort items by descending count
            sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
            
            for item, count in sorted_items:
                if count >= 2:
                    item_index = non_core_items.index(item)
                    highest_level_cluster = find_highest_level_cluster(item_index, all_clusters, set(non_core_items), non_core_items, set(), threshold=0.6)
                    character_clusters.add(tuple(sorted(highest_level_cluster)))
        else:
            character_clusters = set()
        
        results[character] = (builds[0], core_items, character_clusters, duplicate_items)
    
    # For the overall cluster hierarchy, we'll use all items including core items
    all_items = list(set(item for builds in character_builds.values() for build in builds for item in build))
    co_occurrence_matrix, items = create_co_occurrence_matrix(character_builds, all_items)
    similarity_matrix = cosine_similarity(co_occurrence_matrix)
    linkage_matrix = hierarchical_clustering(similarity_matrix)
    cluster_hierarchy = get_cluster_hierarchy(linkage_matrix, all_items)
    
    return results, cluster_hierarchy

def bottom_up_clustering(item_frequencies: Dict[str, int], cluster_hierarchy: List[Tuple[int, List[str]]], n_clusters: int = 6) -> List[Set[str]]:
    # Initialize each item as its own cluster
    clusters = [{item} for item in item_frequencies]
    cluster_frequencies = {frozenset([item]): freq for item, freq in item_frequencies.items()}
    
    # Create a min heap of clusters based on their size, then frequency
    heap = [(1, -freq, frozenset([item])) for item, freq in item_frequencies.items()]
    heapq.heapify(heap)
    
    # Function to find the smallest superset cluster from the hierarchy
    def find_superset_cluster(cluster: Set[str]) -> Set[str]:
        smallest_superset = None
        for _, hierarchy_cluster in cluster_hierarchy:
            hierarchy_set = set(hierarchy_cluster)
            if cluster.issubset(hierarchy_set) and len(hierarchy_set) > len(cluster):
                if smallest_superset is None or len(hierarchy_set) < len(smallest_superset):
                    smallest_superset = hierarchy_set
        return smallest_superset
    
    while len(clusters) > n_clusters:
        if not heap:
            break  # Exit if heap is empty to prevent infinite loop
        
        # Get the smallest cluster
        _, _, smallest_cluster = heapq.heappop(heap)
        smallest_set = set(smallest_cluster)
        
        # Find its smallest superset in the cluster hierarchy
        superset = find_superset_cluster(smallest_set)
        
        if superset:
            # Remove all subsets of the new cluster
            clusters = [c for c in clusters if not c.issubset(superset)]
            heap = [(len(c), -cluster_frequencies[c], c) for _, _, c in heap if not set(c).issubset(superset)]
            
            # Add the new cluster
            clusters.append(superset)
            new_freq = sum(item_frequencies[item] for item in superset)
            cluster_frequencies[frozenset(superset)] = new_freq
            heapq.heappush(heap, (len(superset), -new_freq, frozenset(superset)))
        else:
            # If no superset found, re-add the smallest cluster
            freq = sum(item_frequencies[item] for item in smallest_cluster)
            heapq.heappush(heap, (len(smallest_cluster), -freq, smallest_cluster))

        print(clusters)
        input()
    return clusters

def get_item_frequencies(character_builds: Dict[str, List[List[str]]]) -> Dict[str, int]:
    item_frequencies = defaultdict(int)
    for builds in character_builds.values():
        for build in builds:
            for item in build:
                item_frequencies[item] += 1
    return item_frequencies

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    directory = "data"
    legal_items_file = "all_legal_items.txt"
    normal_items_file = "all_normal_items.txt"

    character_builds = process_directory(directory, legal_items_file, normal_items_file)
    results, cluster_hierarchy = analyze_builds(character_builds)
    item_frequencies = get_item_frequencies(character_builds)
    overall_clusters = bottom_up_clustering(item_frequencies, cluster_hierarchy)

    with open('output.txt', 'w') as f:
        f.write("Character Analysis:\n")
        f.write("===================\n\n")
        for character, (example_build, core_items, clusters, duplicate_items) in results.items():
            f.write(f"Character: {character}\n")
            f.write(f"BIS: {', '.join(example_build)}\n")
            f.write(f"Core Items: {', '.join(core_items)}\n")
            if clusters:
                f.write("Non-Core Item Clusters:\n")
                for cluster in clusters:
                    f.write(f"  {', '.join(cluster)}\n")
            else:
                f.write("No significant non-core item clusters found.\n")
            if duplicate_items:
                f.write("Items that can be built multiple times:\n")
                for item, count in duplicate_items.items():
                    if count == 2:
                        f.write(f"  {item} (twice)\n")
                    elif count == 3:
                        f.write(f"  {item} (thrice)\n")
            f.write("\n")
        
        f.write("\nOverall Cluster Hierarchy:\n")
        f.write("===================\n\n")
        
        f.write("Top 5 non-overlapping clusters:\n")
        for i, cluster in enumerate(overall_clusters, 1):
            sorted_cluster = sorted(cluster, key=lambda x: item_frequencies[x], reverse=True)
            top_six = sorted_cluster[:6]
            f.write(f"{i}. {', '.join(top_six)}")
            f.write("\n")

    print("Results have been saved to output.txt")