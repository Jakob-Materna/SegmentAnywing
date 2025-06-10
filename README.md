# Segment Anywing

[Segment-Anything](https://segment-anything.com/) based segmentation tool created with [Gradio](https://gradio.app/). Select wing cells and calculate cell areas, perimeters wing height and area.

![app-demo](https://github.com/user-attachments/assets/c2899e93-f444-462b-9568-9730df3e0243)

## Installation

Clone this repository：
  ```console
git clone https://github.com/Jakob-Materna/SegmentAnywing.git
  ```

Install the required dependencies using conda and activate the environment:

```
conda env create -f environment.yml
conda activate anywing
```

Create a new directory named `checkpoints` and download the basic [model checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints). Alternativly, you can specify the path to the checkpoint in the `config.yaml`:

  - `vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
  - `vit_l`: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
  - `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (recommended)

Run the tool locally：
  ```
  python app.py
  ```

## Usage

Use positive and negative selection points to generate masks for the wing cells. By default, the cell names correspond to bee wings as described by Charles Michener [1]. The tool can also be adapted for any other type of wing by defining the cell names in the `config.yaml` file.

<img width="318" alt="image" src="https://github.com/user-attachments/assets/f7da6cf9-4d9f-46c9-91ce-ae9792367723" />

## Reference

[1] Michener, Charles D., Robert J. McGinley, and Bryan N. Danforth. The Bee Genera of North and Central America (Hymenoptera: Apoidea). Washington, DC: Smithsonian Institution Press, 1994.

