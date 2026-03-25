# Notes on Physics-Constrained Unmixing

The physical reality of spectral unmixing often involves a very bright signal (e.g., Dye 1) bleeding heavily into another channel (e.g., Dye 2), where the true Dye 2 signal is extremely dim. In this scenario, the image recorded in Channel 2 is almost entirely just Dye 1 bleed-through.

From a mathematical perspective, this physical reality is exactly why the theoretical mixing matrix $M_{theo}$ is the correct bound to use without dynamic scaling.

## The Physical Mixing Model
$$ \text{Image}_1(x, y) = \text{Dye}_1(x, y) + M_{1, 2} \times \text{Dye}_2(x, y) $$
$$ \text{Image}_2(x, y) = M_{2, 1} \times \text{Dye}_1(x, y) + \text{Dye}_2(x, y) $$

### The Problem: Dim Signals Buried in Bleed-Through
Imagine a scenario where Dye 1 is incredibly bright and Dye 2 is incredibly dim, with a bleed-through ratio of Dye 1 into Channel 2 ($M_{2, 1}$) of 10% (0.10).

Because Dye 2 is so dim, it contributes almost nothing to Channel 1:
$$ \text{Image}_1(x, y) \approx \text{Dye}_1(x, y) $$
$$ \text{Image}_2(x, y) \approx 0.10 \times \text{Dye}_1(x, y) + \text{Dye}_2(x, y) $$

In a scatterplot of pixel intensities (Channel 1 on X-axis, Channel 2 on Y-axis), you will see a massive, tight line of pixels with a slope of exactly 0.10.

The Mutual Information optimizer sees this correlation and correctly decides the best way to make the channels independent is to subtract 0.10 times Channel 1 from Channel 2:
$$ \text{Unmixed}_2 = \text{Image}_2 - 0.10 \times \text{Image}_1 $$
$$ \text{Unmixed}_2 = (0.10 \times \text{Dye}_1 + \text{Dye}_2) - 0.10 \times \text{Dye}_1 = \text{Dye}_2 $$
The algorithm successfully recovers the super-dim Dye 2 signal by finding the correct slope (0.10).

### The Real Problem: Biological Co-localization
The problem occurs when Dye 1 and Dye 2 are biologically co-localized (e.g., binding to the exact same structures). If perfectly co-localized, the scatterplot will still be a straight line, but the slope will be steeper. For instance, 0.25 (0.10 from optical bleed-through + 0.15 from actual biological co-localization).

The Mutual Information optimizer is "blind" to biology. It just sees a line with a slope of 0.25 and says, "Let's subtract 0.25 times Channel 1 from Channel 2." If it does this, it subtracts the 0.10 optical bleed-through, but it also hallucinates optical bleed-through and subtracts the real biological Dye 2 signal, resulting in "over-unmixing."

### The Solution: The Theoretical Bound
This is where the theoretical bound $M_{theo}$ saves the day. $M_{theo}$ is computed from the physical spectra and filters. We know mathematically that the laser and filter cannot possibly cause Dye 1 to bleed more than 10% into Channel 2.

We tell the algorithm: "You can optimize Mutual Information all you want, but you are mathematically forbidden from subtracting more than 0.10 times Channel 1." The optimizer tries to subtract 0.25, hits the wall at 0.10, and stops. It successfully removes the 0.10 optical bleed-through and leaves the 0.15 biological co-localization intact.

## The Fallacy of Subtracting "-1.0"
It is common practice to simply subtract the entirety of Channel 1 from Channel 2 (e.g., `Unmixed_2 = Image_2 - 1.0 * Image_1`) to completely erase bleed-through. While this *looks* like it works in image viewers (because negative values are clamped to black), it is mathematically disastrous for co-localized biology.

If Dye 1 is 10,000 photons and Dye 2 is 5,000 photons, and true bleed-through is 10%:
$$ \text{Unmixed}_2 = 5,000 - 0.90 \times 10,000 = -4,000 $$
By subtracting `-1.0` instead of `-0.10`, you didn't just remove the 1,000 photons of bleed-through—you also deleted 4,000 photons of your real Dye 2 signal, creating a massive artificial anti-correlation ("holes" in Channel 2 wherever Channel 1 is bright).

## Evaluating Unmixing with Negative Histograms (The "Shift to Negative" Metric)
Because fluorescence is physically additive (you cannot emit negative photons), a common intuition is to use a strict "Non-Negative" constraint: *stop unmixing the moment any pixel drops below zero.*

This fails in reality because of **camera read noise and shot noise**. Even if an algorithm perfectly removes bleed-through leaving a true signal of `0.0`, camera noise creates a Gaussian distribution around zero (e.g., `-5, 3, 0, -2`). Therefore, in a perfectly unmixed image, **half of the background pixels will naturally be negative.**

If you plot a histogram of pixel intensities for Channel 2 after unmixing, you can visually evaluate the success:
1. **Healthy Unmixing (e.g., -0.75 bounded by physics):** The peak (mode) of the background distribution is pinned tightly to `0`. The negative values form a steep, symmetric drop-off representing only the left tail of the camera's Gaussian noise floor.
2. **Over-Unmixing (e.g., forced -1.0):** The entire distribution shifts massively to the left. The peak is no longer at `0`, and a huge chunk of pixels is shoved deep into the negatives. This "fat negative tail" represents real biological structure that was erroneously subtracted away.

Thus, we **can** use the shift to negative as a metric, provided we look at the *distribution* (ensuring the mode remains at 0 and the negative tail matches the noise profile) rather than preventing individual noisy pixels from dropping below zero.

## Synthetic Stress Tests (In Silico)
To verify this behavior, we built synthetic stress tests:

1. **The "Floodlight and Candle" Test (SNR limits)**
   - Setup: Synthetic image where Dye 1 has intensities ~50,000 and Dye 2 ~100. Mix them with 10% bleed-through. Add Poisson noise.
   - Result: The shot noise of the 5,000-photon bleed-through is around $\pm70$ photons, completely swallowing the 100-photon Dye 2 signal. The unmixing algorithm fails to find the correlation because the signal-to-noise ratio is too low. This test intentionally fails (`xfail`), showing the practical mathematical limits of recovery in the presence of physical noise.
2. **The Perfect Co-localization Trap (Tests over-unmixing)**
   - Setup: Dye 1 and Dye 2 are precisely the same shape (perfectly co-localized) and brightness. Mixed with 10% optical bleed-through.
   - Result: The unconstrained empirical unmixing algorithm over-unmixes because it attributes the biological correlation to optical bleed-through. The physics-constrained approach hits the $M_{theo}$ wall at 10% exactly and successfully stops, preserving the biological signal.
3. **The "Bad Physics" Test (Tests sensitivity to theory)**
   - Setup: Image mixed with true 15% bleed-through. The unmixing algorithm is fed an incorrect $M_{theo}$ claiming the max bleed-through is only 5%.
   - Result: The physics-constrained algorithm hits the 5% wall and stops, resulting in under-unmixing. This demonstrates the pipeline's strict obedience and sensitivity to the provided theoretical metadata (e.g., from FPbase).