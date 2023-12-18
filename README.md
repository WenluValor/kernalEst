# Readme
## File Introduction
* mse_rep.py (main program): including point and interval estimation. For now, only point estimation is ran. Interval estimation remains to change (the codes below regress_boots())

* data_1.py, data_2.py, data_3.py: generate data and provide $f_0$ for evaluation.

* main.py: regress on $\hat{f}_n$.

* kernel.py: use multiple Matern kernel to get the random mapping features.

* test_1.py, test_2.py, test_3.py: calculate MSEs and STDs from point estimation.

## How to produce MSEs and STDs
To produce MSEs and STDs, you only need to edit boots.py, main.py, test_x.py (test file depends on which case you are looking for).
1. Set proper paramters in mse_rep.py, and create a folder 'MSE_outcome' under the directory.
   * $s$ (dimensions of random features): can be any positive integer.
   * $n$ (number of observations): decide how many data points will be generated.
   * $d$ (dimensions of data): $d=3$ for all three examples.
   * $p$ (type of gradients): must $\le d$.
   * $B$ (number of bootstrap): we take $B=200$ for all three examples.
   * case_num (type of examples): can be either $1, 2, 3$, allowing for different data-generation process.
2. Data generation in boots.py.
   
   Set the proper data generation parameters in the function **dt_x.generate_data()**, remember to remove the annotation sign if it blocks data generation.

   It will produce vec_t.csv, Y0.csv, Z.csv, whose meanings are $\boldsymbol{t}_i^{(j)}$, $Y^{(0)}$, $Y^{(j)}$ in the paper.

4. Find lengthscale of kernel.

   Keep **knl.generate_wb(d_value=d, s_value=s)** in case it is annotated.

   It will produce tau.csv, w_js.csv, b_js.csv, whose meanings are $\tau$, $w$, $b$ in the paper.

6. Create kernel matrix.

   Keep **main.create_mat(s=s, N=n, d=d, p=p)** in case it is annotated.

   It will produce mat_PSI.csv, part_mat_PSI.csv, whose meanings are $\Psi_{p(d+1)}(\boldsymbol{t}_i^{(j)})$, $\frac{\partial }{\partial t_j} \Psi$ in the paper.

8. Regress to find $\boldsymbol{c}$.
   
   Keep **main.boots_compute(N=n, s=s, d=d, p=p, use_lmd=False)** in case it is annotated.

   It will produce torch_C.csv, whose meaning is $\boldsymbol{c}$ in the paper.

   Default is not to use $\lambda$ for regularization, since it turns out to be a better result. If you set **use_lmd=True**, we also find the minimizer $\lambda$ for GCV, doing so will create the file lmd.csv.

10. Set proper learning rates and tolerance:

    Before clicking on 'run', you should check the learning rate and tolerance in line 145 of main.py, which is the last line inside the function **get_vec_c_torch()**.

    The regression relies highly on the random features, so we recommend tunning for different learning rate and tolerance based on what you derive from kernel.py. In generation, a recommendation for learing rate and tolerance are:
    * Case1: (tol, lr)= $(1e^{-4}, 1e^{-2})$.
    * Case2: (tol, lr)= $(1e^{-8}, 1e^{-2})$.
    * Case3: (tol, lr)= $(1e^{-12}, 1e^{-3})$.

11. Click 'run' in the mse_rep.py.
12. Evaluate MSEs and STDs:
    
    Run the test_x.py after setting the correct $s, n, d, p$ in the **set_global()**.

    The program will print $200$ times of MSEs, and print the avarage MSEs and STDs among the $200$ evaluations in the end.


