# What this taught me

I trained the one-layer transformer from Nanda et al. on modular addition mod 113: 30% of the data, weight decay 1.0, one seed, 50,000 epochs, about 46 minutes on my Mac. The paper's central phenomenon showed up in this run: the model memorized the training set early, test accuracy sat near zero for thousands of epochs, then climbed to near-perfect between roughly epochs 8,600 and 9,600. The mechanistic checks the paper describes also held when I applied my own implementations of its analyses to my weights. That is one seed plus independent reimplementations of the paper's analyses, not the whole paper reproduced.

## Watching the circuit assemble before accuracy moves

The most useful thing the progress measures taught me: test accuracy is a lagging indicator. Through the long plateau, where the model looks like a lookup table, the loss through just the key frequencies keeps falling the entire time. Something that generalizes is assembling while the headline number refuses to move. Over the same stretch, the loss through all the remaining frequencies rises and peaks late in the plateau. One reading of that pair is a handover: a memorizing component carries the training set while a structured component matures underneath, and grokking is the moment the second takes over. That is an interpretation of two curves, not a proven mechanism, but it changed what a flat validation curve means to me.

## The Fourier structure is real, not a plotting device

The analysis found five key frequencies, the same count the paper reports for its analyzed model, and four of the five even match its set (14, 35, 42 and 52; the paper's fifth is 41, mine is 15). The specific set is a property of the trained weights, not a spec the model had to hit. What convinced me was the ablation, not the plot: keep only the five key frequencies and the model still solves the task; remove them and it lands worse than guessing uniformly. Cutting any single key frequency hurts badly; cutting non-key frequencies costs little by comparison. I would not claim the five frequencies are the whole algorithm. What this run supports is narrower: on this seed, the generalizing solution leans almost entirely on them.

## One seed has texture that averages hide

The paper's curves average five seeds; mine is one trajectory and looks like one. Test loss spiked hard early, and after the main cleanup the Gini coefficient and weight norm kept oscillating instead of settling flat, even though their broad shapes match the paper's description. I cannot claim those differences are pure seed noise, only that this run does not contradict the averaged picture. Same caution on reproducibility: two short pilots with the same seed drifted apart within a few hundred epochs on my machine, where the MPS backend does not guarantee deterministic ops. I cannot prove the backend caused the drift, but I also cannot claim a second full run would retrace this one.

## What I take into the next one

A flat validation curve does not mean nothing is happening; the interesting structure can be forming underneath. Single-seed runs deserve single-seed language: say what this trajectory did, not what the phenomenon does. And a figure has to survive its own scale: my ablation chart on a linear axis hides the small bars completely, so the plot meant to carry the argument was the one that read worst.
