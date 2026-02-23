# 3-Minute Presentation Script

**Time Limit:** 3 Minutes Talk / 2 Minutes Q&A

---

## Slide 1: The Bottleneck (0:00 - 0:45)
**"Imagine you own a small factory.** Every day, you have thousands of products rolling down the line.
Right now, your only way to check for quality is **manual inspection**. You have valid people staring at water bottles for 8 hours straight.
It’s exhausting. Eyes get tired. Attention slips. And inevitably, **defects get through**.
One bad bottle reaches a customer, and your reputation takes a hit. This is the bottleneck for every small manufacturer."

---

## Slide 2: The Cost Barrier (0:45 - 1:30)
**"So, you look for a solution.** You find these incredible 'Vision Systems' on the market.
They work great. But then you see the price tag: **$18,000**.
For a giant like Coca-Cola, that’s nothing. But for a small family-owned factory? That’s your entire profit margin for the month.
You are stuck. You’re too small for the big tech, but you’re too big to keep doing things by hand."

---

## Slide 3: The Complexity Trap (1:30 - 2:15)
**"Then someone tells you: 'Just use AI!'**
But here is the trap no one talks about. Standard AI is **expensive in a different way**: Time Complexity.
To make a traditional AI work, you need:
1.  Thousands of images of 'bad' products (which you hopefully don't have yet!).
2.  Someone to sit there and manually label every crack and scratch.
3.  Days of training time.
Small factories don't have Data Science teams. They need something that works **now**."

---

## Slide 4: The Solution (2:15 - 3:00)
**"This is why I built 'Zero-Training Quality Control'.**
My solution uses something called **One-Shot Learning**.
Here is the magic: I don't need thousands of bad examples.
**I just need ONE picture of a GOOD product.**
You show the system *one* perfect bottle. It learns the geometry, the symmetry, the look.
Then, anything that *doesn't* match that one image is instantly flagged as a defect.
**No training time. No data labeling. Just instant quality control.**

To evaluate this, I run the model on a dataset of test images.
For each image, the model assigns a precise **Anomaly Score**.
If this score crosses a specific **learned threshold**, the system instantly flags it as a defect.
It also generates a **Heatmap** to visualize exactly *where* the defect is on the bottle.

Next week, I will demonstrate this live. Thank you."

---

# Q&A Cheat Sheet (2 Minutes)

**Q: How will you evaluate success?**
*A: "I'm focusing on three metrics: Accuracy (aiming for >90%), False Positive Rate (keeping it low so we don't throw away good products), and Latency (must run in real-time)."*

**Q: How is this different from standard anomaly detection?**
*A: "Standard methods often strictly compare pixels or statistical outliers. My approach uses a CNN to understand 'structural' features, so it's more robust to minor lighting changes than simple pixel subtraction."*

**Q: How do you train the AI for one-shot detection?**
*A: "We don't 'train' weights from scratch. We use a pre-trained wide-ResNet to extract features. Then, we fit a 'Memory Bank' of good features from the one reference image. The AI simply compares new images to this memorized bank of 'perfect' features."*

**Q: How does it handle different lighting?**
*A: "Great question. We use standard image processing to normalize brightness, but for the best results, we recommend a simple consistent light source, like a ring light."*

**Q: What if the product rotates?**
*A: "The system is designed to check for symmetry and features regardless of minor rotation, but we can also align the image digitally before analysis."*

**Q: Is it as accurate as the $18,000 machine?**
*A: "For the specific defects we target—like cracks or missing labels—yes. It might not measure to the micrometer, but for spotting 'good vs bad,' it is highly effective and a fraction of the cost."*
