## Need polygon structure

```python
raycasting algorithm
def create_binary_mask(polygons, width, height, device=None):
    """
    Create a binary mask where points inside any of the polygons are marked as 1 and others as 0.

    Args:
        polygons: A list of polygons, where each polygon is represented by its vertex coordinates
                    with shape (n, 2), where n is the number of vertices.
        width: Width of the plane.
        height: Height of the plane.

    Returns:
        A binary mask with shape (height, width), where points inside any of the polygons are marked as 1 and others as 0.
    """
    # Determine the device to use
    if device is None:
        device = polygons[0].device
    # Create a grid representing all points on the plane
    x = torch.arange(width, dtype=torch.float32, device=device)
    y = torch.arange(height, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    points = torch.stack([grid_x, grid_y], dim=-1)
    # Initialize the binary mask
    mask = torch.zeros_like(grid_x, dtype=torch.bool)

    # for polygon in polygons:
        # Ensure the input data is a PyTorch tensor
        # Convert polygon vertices to edge representation
        # edges = torch.cat([polygon, polygon[:1]], dim=0)


    # Extract all edges from the polygons
    edges = []
    for polygon in polygons:
        edges.append(torch.cat([polygon, polygon[:1]], dim=0))
    edges = torch.cat(edges, dim=0)

    # Initialize the intersection counter
    count = torch.zeros_like(grid_x, dtype=torch.int32)

    for i in range(len(edges) - 1):
        # Calculate the vectors from each point to the edge endpoints
        v1 = edges[i] - points
        v2 = edges[i + 1] - points
        # Calculate the cross product of v1 and v2
        cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        # Check if the point is on the edge
        on_edge = (v1[..., 0] == 0) & (v1[..., 1] >= 0) & (v2[..., 1] < 0)
        # Check if the point is inside the polygon
        inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | (
            (v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0)
        )
        # Increment the count for points inside the polygon or on the edge
        count += (inside | on_edge).int()
    # If the count is odd, the point is inside the current polygon
    print(torch.max(count))
    mask |= count % 2 == 1
    return mask
```

______________________________________________________________________

Non-polygon structure,
take out all edges and calculate at once.
