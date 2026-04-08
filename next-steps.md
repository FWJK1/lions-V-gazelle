1. Write and run function that 
    * iterates over `results` and each experiment / terminal / breeding strategy folder
    * checks if animations have been saved with the same file structure in `animation` (best positions and random positions)
    * if animation is not present, create and save animation. 

2. rewrite `results.qmd`
    *  load in animations, not create them 
    * take `id` as a parameter

3. rerun quarto for each experiment. If any fail / have insufficient data, write a new toml to get just that data and run it on VACC, then run it. 

4. Assess difference, potentially also add stuff 

5. Put comparative results into a new `results-summary.qmd`