# Plant Health Model — Android Integration

Binary image classifier: is the plant in this photo healthy or unhealthy?

Everything in this folder is what you need. You do not need the training code, the dataset, or Python.

---

## Files

| File | What it is |
| --- | --- |
| `model.tflite` | The model. Float16 quantised, 4.51 MB. Drop into `app/src/main/assets/`. |
| `labels.txt` | Class names in index order: `Healthy`, `Unhealthy`. |
| `test_vectors/` | Four real images plus `expected.txt` with the exact output each should produce. |

---

## Input contract

```
shape:  [1, 160, 160, 3]
dtype:  float32
layout: RGB
range:  0.0 to 255.0
```

### Do not normalise the pixels

This is the one thing that will silently break the integration.

The model has a rescaling layer built into it. It converts 0–255 into the range MobileNetV2 expects, internally. So it needs **raw pixel values**.

```kotlin
// CORRECT
val value = ((pixel shr 16) and 0xFF).toFloat()   // 0..255

// WRONG - the model will do this again internally
val value = ((pixel shr 16) and 0xFF) / 255.0f
```

If you normalise, there is no crash and no error message. The model just returns meaningless results. If accuracy looks close to random, this is the first thing to check.

Resize the camera image to exactly 160x160 before feeding it in. Bilinear resize is what the model was trained with.

---

## Output contract

```
shape: [1, 1]
dtype: float32
```

One number between 0 and 1, the probability that the plant is **unhealthy**.

```kotlin
val unhealthy = output[0][0] > 0.5f
val confidence = if (unhealthy) output[0][0] else 1f - output[0][0]
```

| Output | Meaning |
| --- | --- |
| near 0.0 | confidently Healthy |
| near 0.5 | uncertain |
| near 1.0 | confidently Unhealthy |

Class indices are alphabetical, so index 0 is Healthy and index 1 is Unhealthy. That ordering matches `labels.txt`.

### About the 0.5 threshold

0.5 is the neutral default, and the reported accuracy uses it. It is a product decision, not a fixed property of the model. Lowering it to, say, 0.3 makes the app flag more plants as unhealthy: it catches more real disease but raises false alarms. Talk to us before changing it, because our published numbers assume 0.5.

---

## Verifying your integration

Run the four images in `test_vectors/` through your code. You should get these numbers, give or take about 0.01 for resize and colour conversion differences:

| Image | True class | Expected output |
| --- | --- | --- |
| `healthy__RS_HL 7544.JPG` | Healthy | 0.000002 |
| `healthy__RS_HL 5941.JPG` | Healthy | 0.013255 |
| `unhealthy__FREC_Scab 3335.JPG` | Unhealthy | 0.999992 |
| `unhealthy__FREC_Scab 3504.JPG` | Unhealthy | 0.998636 |

If your numbers are wildly different, especially if they all sit near 0.5, check the normalisation issue above first, then check RGB versus BGR channel order.

---

## Performance

Measured on a desktop CPU, so treat these as a rough guide rather than phone numbers:

- 314 images per second batched
- Model file 4.51 MB

MobileNetV2 was designed for mobile, so single image inference should be comfortably real time on a modern phone. If you need more speed, the GPU delegate works with this model, since it is float16.

---

## Accuracy

Measured on 8147 test images the model never saw during training:

| Class | Precision | Recall | F1 | Images |
| --- | --- | --- | --- | --- |
| Healthy | 0.9890 | 0.9947 | 0.9918 | 2263 |
| Unhealthy | 0.9980 | 0.9958 | 0.9969 | 5884 |

Overall accuracy 99.55%, which is 37 mistakes out of 8147. Of those, 25 were diseased plants called healthy and 12 were healthy plants flagged as diseased.

These are the numbers for the `.tflite` file in this folder, not just the original model. We checked the converted model separately and it produces identical predictions.

---

## Important limitation, please read

The model was trained on the PlantVillage dataset, where every leaf is photographed **against a plain, uniform background** in good lighting.

Real photos from a phone camera, with soil, other leaves, shadows and clutter in frame, are meaningfully different from anything the model has seen. Expect real world accuracy to be **lower than 99.55%**, possibly noticeably so.

This is not a bug in the model or in your integration. It is a known limitation of the training data.

What helps:
- Guide the user to fill the frame with a single leaf
- Prefer a plain background if the UI can suggest one, for example a hand or a sheet of paper behind the leaf
- Show the confidence value rather than a bare yes or no, so a borderline result looks borderline

If you can collect real photos through the app, send them back to us. Retraining on real images is the single biggest improvement available.

---

## Scope

The model answers healthy or unhealthy only. It does **not** identify which disease, and it does not identify the plant species. It was trained on crop leaves such as tomato, apple, grape, corn and potato. Photos of something else entirely, for example a person or a wall, will still produce a confident looking number, because the model has no way to say "this is not a leaf". Handle that in the UI if it matters.

---

Questions, or numbers that do not match the test vectors: come back to us before working around it.
