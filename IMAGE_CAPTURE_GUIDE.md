# Image Acquisition Guidelines for G-CNN Defect Detection

## Why Consistency Matters

The G-CNN model learns the appearance of normal products during training. Any variation in imaging conditions (angle, distance, lighting) between training and test images will be detected as anomalies, leading to false positives.

## Recommended Setup

### 1. Camera Position
- **Distance**: Fixed distance from product (e.g., 50cm)
- **Angle**: Straight-on, perpendicular to product
- **Height**: Camera at product center height
- **Consistency**: Use a tripod or fixed mount

### 2. Product Placement
- **Background**: Plain, uniform color (white or gray recommended)
- **Position**: Product centered in frame
- **Orientation**: Same rotation for all images
- **Stability**: Product should not move during capture

### 3. Lighting
- **Type**: Diffuse, even lighting (avoid harsh shadows)
- **Position**: Front or top lighting, not side lighting
- **Consistency**: Same lighting setup for all images
- **Avoid**: Direct sunlight, reflections, glare

### 4. Frame Composition
- **Coverage**: Entire product visible in frame
- **Margins**: 10-20% border around product
- **Avoid**: Cutting off parts of product (top, bottom, sides)
- **Focus**: Product in sharp focus

### 5. Image Quality
- **Resolution**: At least 1024x1024 pixels
- **Format**: JPEG or PNG
- **No filters**: No Instagram filters, color adjustments, etc.
- **Stable camera**: Avoid motion blur

## Recommended Workflow

### For Training Data (Normal Products)
1. Set up camera with tripod at fixed position
2. Mark product placement spot on surface
3. Set fixed lighting
4. Capture 50-100 images of different normal products
5. Keep exact same setup for all images

### For Test Data
1. Use SAME camera position as training
2. Use SAME product placement
3. Use SAME lighting conditions
4. Take photos of both normal and defective products
5. Label each image appropriately

## Current Issue

From your screenshot, the bottle is too close to the camera and cut off at the bottom. This creates two problems:
1. Model cannot analyze the full product
2. Different framing than training data causes false anomalies

## Recommended Fix

1. **Take new photos** with entire bottle visible in frame
2. **Maintain consistency**: Same angle, distance, lighting for ALL images
3. **Include margins**: Leave 10-20% space around bottle
4. **Retrain or recalibrate**: After capturing consistent images

## Quick Checklist

Before taking each photo:
- [ ] Tripod/camera position unchanged
- [ ] Product centered in viewfinder
- [ ] Entire product visible (no parts cut off)
- [ ] Lighting consistent
- [ ] Background clear
- [ ] Product in focus
- [ ] Same orientation as training images

## Example Setup

```
Camera (tripod-mounted, 50cm away)
         |
         v
    [Background]
    [  Product  ]
    [   Table   ]
         ^
    Marked spot
```

## After Recapture

Once you have consistent images:
1. Replace images in `data/water_bottles/train/good/`
2. Replace images in `data/water_bottles/test/`
3. Retrain model or recalibrate threshold
4. Run evaluation

This will significantly improve model accuracy and reliability.
