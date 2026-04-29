## Notes

- [ ] The SRAF is not useful, should add the original version.

- [ ] Do some test on the best parameter set.

  - [ ] Optmizer.

Apr 20:

- [ ] Baseline results organize.

  - [ ] Use the original center test.

- [ ] Exp on larger cases.

- [ ] check back on EPE loss of the best parameters.

Apr 19:

When the resolution is lower than 256, no SRAF will be generated.

- [x] First round gen SRAF candidates.
- [x] Second round DiffOPC
- [x] Exp on SRAF min threshold
- [x] Exp on SRAF initial wh
- [x] Load SRAF in the binary function

______________________________________________________________________

Apr 18

- [x] EPE loss is not good.
  - [x] Do more epe loss exp.
  - [x] Ablation on epe parameters.
- [x] SRAF generation on low resolution region
- [x] use pre-generated SRAF to compute.

Apr 14

- [x] Add exp for non-kernel curv.
- [x] Add exp for stepsize

edges -> vertices (must have , but vertices can be smaller)

Apr 12

- [x] scipy-gpu / opencv-gpu rasterization
- [x] mrc aware

Apr 11

- [x] rasteration has bug.
- [x] The downscale didn't make it faster.
- [x] Should calculate the baseline earlier.

Apr 10

Multi-level EdgeILT

- [x] use 512 1024 2048 three levels
- [x] when SRAF inseration?
  - [x] reduce the SRAF size, once the mid is out of the generation region.
  - [x] mask save the binary image and not moving.
  - [x] how to remove some small isolated points.
- [x] How many round of low-resolution?
- [x] How many round of high-resolution?
- [x] How to make sure all the edges can be connected with each other?
- [x] add curv to reduce the noise points?
- [x] Test SGD

baseline

- [x] Multi-level edge-ILT
- [x] STEP resize on the velocity vector.
- [x] EPE loss
- [x] comparison with previous code

check when to add sraf

- [x] can add a larger forbidden mask at the beginning.

______________________________________________________________________

Apr 10

1. Separated region algorithm for raycasting is more accurate, but not faster.
2. Faster than the whole region algorithm.

Apr 8

- [x] SRAF insertion algorithm

  - [x] Forbidden region
  - [x] Get the maximum
  - [x] Use the red region
  - [x] Will be more stable after 70 epoch
  - [x] Once the gradient drop below 0.6

- [x] SRAF parameters

  - [x] minimim area
  - [x] min_threshold
  - [x] min_sraf_seed number

SRAF candidates:

1. Outside the forbidden region

[contourf algo](https://zj-image-processing.readthedocs.io/zh-cn/latest/matplotlib/%E7%AD%89%E9%AB%98%E7%BA%BF%E5%9B%BE/)

## [marching square algorithm: github](https://github.com/nathanbreitsch/convsquares/blob/main/marching_squares.py)

Apr 1st

- case 4 wrong
- case 10 wrong

Add Offset region to accelerate the computation.

______________________________________________________________________

Some cases that the merge corner will fail.

Mar 31

- [x] SRAF insertion algorithm

- [x] logging system

- [x] Debug why sometimes the edge can not be merged

- [x] Accelerate through offset region

- [x] organize codes, add logging system

- \[ \]

## The special corner

```
└┐
```

## Mar 29

- [x] debug why it lose the rect shape
- [x] debug the vel vector
- [x] add video combiner
- [x] optimize the raycasting algorithm
- [x] test on more cases

## Raycasting-based algorithm

The smallest set to recovery a mask

1. edge segments.
2. edge direction vector.
3. polygon ids.
4. metadata: image width / height

## Tree construction

V-edges: easy to get linked list

H-edges: easy to get linked list

start
end
next

```python
            segments.append(
                {
                    "segment": torch.stack([start_point, end_point], dim=1),
                    "type": seg_type_label,
                    "id": segment_id,
                    "start": False,
                    "end": True,
                    "next": "To next edge"
                }
            )
```

The segments structure.

```
->->->->|
|       |
|       |
|->->->->
```

______________________________________________________________________

LevelSet

```python
mask = self._binarize(params)
printedNom, printedMax, printedMin = self._lithosim(mask)
```

______________________________________________________________________

Todo

seg_params to polygon and binary images

```

    all_polygons_by_all_id = []
    for polygon in all_polygons_by_start_id:
        polygon_all_ids = []
        for seg_id in polygon:
            edge_list = get_sub_by_start(all_traverse_list, seg_id)
            polygon_all_ids.extend(edge_list)

```

```

    all_polygons_by_start_id = [[element for tuple_pair in sublist for element in tuple_pair] for sublist in all_polygons_by_start_pair]
    print(all_polygons_by_start_id)



```

```python
        vertices = torch.tensor([[574., 478.],
[574., 802.],
[706., 802.],
[706., 738.],
[638., 738.],
[638., 478.]]).unsqueeze(0)
```

Use the right sequence, it can be good.

______________________________________________________________________

New solution

When construct the polygon, get to the edge.
Record the relevant information.

______________________________________________________________________

```text
BEGIN     /* GL1TOGULP CALLED ON FRI MAY 17 11:33:25 2013 */
EQUIV  1  1000  MICRON  +X,+Y
CNAME Temp_Top
LEVEL M1

CELL Temp_Top PRIME
   RECT N M1  80  80  252  126
   RECT N M1  80  250  256  126
ENDMSG
```

polygon

```python
[[[80, 80], [80, 206], [332, 206], [332, 80]], [[80, 250], [80, 376], [336, 376], [336, 250]]]
```

reshape

```
[[[128, 108], [128, 234], [380, 234], [380, 108]], [[128, 278], [128, 404], [384, 404], [384, 278]]]
```

______________________________________________________________________

test polygon

```python
vertices = torch.tensor(
    [
        [200, 200],
        [900, 200],
        [900, 900],
        [200, 900],
    ],
    dtype=torch.float32,
    device=DEVICE,
)
```

______________________________________________________________________

这是一个tensor表示的两条边：

```python
[[[ 680.,  680.],          [1046., 1070.]],          [[ 680.,  680.],          [1070., 1110.]]]
```

______________________________________________________________________

edge 的向量表示 \[N, 2, 2\] ,

N个edge，2：起点和终点，2：2-D（x,y)

```python
[[x1,x2],[y1,y2]]
```

first edge:

```python
tensor([[ 680.,  680.],
[1070., 1110.]], device='cuda:0', grad_fn=<SelectBackward0>)
dir: tensor([0., 1.], device='cuda:0')
vel: tensor([1., -0.], device='cuda:0')
```

______________________________________________________________________

Better edge view

```python
    print("edges")
    edge_view = edges.clone().detach().transpose(1, 2)
    print(edge_view)
```
