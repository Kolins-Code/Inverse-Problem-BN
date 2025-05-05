using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

using System.IO;
using System.Globalization;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Factors;
using System.Data;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Collections;


class ModelLaplace3
{
    /*static Variable<double> norm_dist(Variable<double> x)
    {
        return 1.0 / (Math.Sqrt(2.0 * Math.PI) * 0.1) * Variable.Exp(-1 * ((x - 0.5) / 0.1) * ((x - 0.5) / 0.1) / 2);
    }*/

    static double norm_dist(double x)
    {
        return 1.0 / (Math.Sqrt(2.0 * Math.PI) * 0.04) * Math.Exp(-1 * ((x - 0.5) / 0.04) * ((x - 0.5) / 0.04) / 2);
    }


    static void Main(string[] args)
    {
        //args = new[]{"0.0001", "0.01", "D:/Programming/InverseProblemBN/Runs/run 19.11.24 - 11.27.49/input/data.csv", "D:/Programming/InverseProblemBN/Runs/run 19.11.24 - 11.27.49/output/data.csv"};
        var culture_param = new CultureInfo("en-EN");
       
        double dx = Convert.ToDouble(args[0], culture_param);
        double end_x = Convert.ToDouble(args[1], culture_param);

        int size_x = (int) (end_x / dx);

        var reader = new StreamReader(args[2]);

        double[] function1_obs = new double[size_x + 1];

        int k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            function1_obs[k] = Convert.ToDouble(line, culture_param);
            k++;
        }
        reader.Close();
        reader = new StreamReader(args[4]);

        double[] derivative_function1_obs = new double[size_x + 1];

        k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            derivative_function1_obs[k] = Convert.ToDouble(line, culture_param);
            k++;
        }
        reader.Close();
        reader = new StreamReader(args[3]);

        double[] function2_obs = new double[size_x + 1];

        k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            function2_obs[k] = Convert.ToDouble(line, culture_param);
            k++;
        }
        reader.Close();
        reader = new StreamReader(args[5]);

        double[] derivative_function2_obs = new double[size_x + 1];

        k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            derivative_function2_obs[k] = Convert.ToDouble(line, culture_param);
            k++;
        }
        reader.Close();

        double[] function_t_obs = new double[size_x + 1];
        for (int i = 0; i < size_x + 1; i++)
        {
            function_t_obs[i] = -Math.Abs(i*dx - end_x / 2) + end_x / 2;
        }
        

        Range grid_x = new Range(size_x + 1).Named("Сетка по x");
        Range grid_t = grid_x.Clone();

        Range grid_3x = new Range((size_x + 1) * 3);
        
        VariableArray<VariableArray<double>,double[][]> solution = Variable.Array(Variable.Array<double>(grid_x), grid_t).Named("Решение прямой задачи");
        VariableArray<double> end_function = Variable.Observed(function2_obs);
        VariableArray<double> start_function = Variable.Observed(function1_obs);
        /*VariableArray<double> derivative_start_function = Variable.Array<double>(grid_x);
        VariableArray<double> derivative_end_function = Variable.Array<double>(grid_x);*/
        VariableArray<double> derivative_start_function = Variable.Observed(derivative_function1_obs);
        VariableArray<double> derivative_end_function = Variable.Observed(derivative_function2_obs);

        VariableArray<double> function_t = Variable.Observed(function_t_obs);

        double[] shifts_obs = new double[size_x + 1];
        for (int i = 0; i <= size_x; i++)
        {
            shifts_obs[i] = i * dx;
        }
        VariableArray<double> shifts = Variable.Observed(shifts_obs);

        double[] norm_dist_obs = new double[size_x * 3 + 1];
        for (int i = 0; i < size_x * 3 + 1; i++)
        {
            norm_dist_obs[i] = norm_dist(-1 + i*dx);
        }
        VariableArray<double> norm_dist_var = Variable.Observed(norm_dist_obs);
        //VariableArray<double> function_tt = Variable.Observed(function_t_obs);

        /*using (ForEachBlock x = Variable.ForEach(grid_x))
        {  
            derivative_start_function[x.Index] = Variable.GaussianFromMeanAndVariance(derivative_start_function_mean[x.Index], 0.1); 
            derivative_end_function[x.Index] = Variable.GaussianFromMeanAndVariance(derivative_end_function_mean[x.Index], 0.1); 
        }*/

        VariableArray<double> function = Variable.Array<double>(grid_x);
        VariableArray<double> function_mean = Variable.Array<double>(grid_x);
        //VariableArray<double> function_prec = Variable.Array<double>(grid_x);
        function_mean[grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_x);
        //function_prec[grid_x] = Variable.GammaFromShapeAndScale(1, 1).ForEach(grid_x);
        function[grid_x] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(grid_x);;
        

        VariableArray<double> noise_prec = Variable.Array<double>(grid_x);
        noise_prec[grid_x] = Variable.GammaFromShapeAndScale(1, 1).ForEach(grid_x);
        VariableArray<double> noise = Variable.Array<double>(grid_x);
        noise[grid_x] = Variable.GaussianFromMeanAndPrecision(0, 100).ForEach(grid_x);
        Variable<double> n = Variable.GaussianFromMeanAndVariance(0, 0.01);
        //noise[grid_x] = Variable.GaussianFromMeanAndPrecision(0, noise_prec[grid_x]);

        Range grid_xx = new Range(size_x).Named("Сетка по x без одного");
        Range grid_tt = grid_xx.Clone();

        Range grid_xxx = new Range(size_x - 1);

        /*double[] probs = new double[size_x + 1];
        for (int i = 0; i <= size_x; i++)
        {
            probs[i] = 1 / (size_x + 1);
        } 
        Variable<int> index = Variable.Discrete(grid_x, probs);*/
        Variable<double> index_x = Variable.GaussianFromMeanAndVariance(0, 100);

        Variable<double> c1 = Variable.GaussianFromMeanAndVariance(0, 100);
        Variable<double> c2 = Variable.GaussianFromMeanAndVariance(0, 100);
        Variable<double> c3 = Variable.GaussianFromMeanAndVariance(0, 100);

        Variable<double> c1_prec = Variable.GammaFromShapeAndScale(1, 1);
        Variable<double> c2_prec = Variable.GammaFromShapeAndScale(1, 1);
        Variable<double> c3_prec = Variable.GammaFromShapeAndScale(1, 1);

        using (ForEachBlock t = Variable.ForEach(grid_t))
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {  
                solution[t.Index][x.Index] = Variable.GaussianFromMeanAndVariance(0, 0.01);
            } 
            //function_t[t.Index] = Variable.GaussianFromMeanAndVariance(function_tt[t.Index], 0.0001); 
        }
        /*using (ForEachBlock x = Variable.ForEach(grid_x))
        {  
            solution[0][x.Index] = Variable.GaussianFromMeanAndPrecision(0, 0.01); 
        }*/
        

        //solution[1][0] = 0;
        //solution[1][size_x] = 0;

        /*using (ForEachBlock t_block = Variable.ForEach(grid_t))
        {
            var t = t_block.Index;
            using (ForEachBlock x_block = Variable.ForEach(grid_x))
            {
                var x = x_block.Index;
                var bounds = x == 0 & x == size_x;
                using (Variable.If(t == 0))
                {
                    solution[t][x] = Variable.GaussianFromMeanAndPrecision(0, 0.01); 
                }
                using (Variable.If(bounds))
                {
                    solution[t][x] = 0;
                }
                using (Variable.If(!bounds & t == 1))
                {
                    solution[t][x] = (-solution[t - 1][x - 1] - solution[t - 1][x + 1] + 4 * solution[t - 1][x] + 2 * dx * derivative_start_function[x] + function_mean[x] * dx) / 2;
                }
                using (Variable.If(!bounds & t > 1))
                {
                    solution[t][x] = -solution[t-2][x] + 4 * solution[t-1][x] - solution[t-1][x - 1] - solution[t-1][x + 1] + function_mean[x] * dx * dx;
                }

            }
            
            //Variable.ConstrainEqual(solution[t][size_x], 0);
           //solution[t][size_x] = 0;
        }*/
        /*solution[0][grid_x] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(grid_x);

        using (ForEachBlock x_block = Variable.ForEach(grid_x))
        {
            var x = x_block.Index;
            var bounds_cond = x == 0 | x == size_x; 
            
            using (Variable.If(x == 0)) 
            {
                solution[1][x] = 0;
            }
            using (Variable.If(x > 0)) 
            {
                solution[1][x] = (-solution[0][x - 1] - solution[0][x + 1] + 4 * solution[0][x] + 2 * dx * derivative_start_function[x] + function_mean[x] * dx) / 2;
            }
        }
        for (int t = 2; t <= size_x; t++)
        {
            using (ForEachBlock x_block = Variable.ForEach(grid_x))
            {
                var x = x_block.Index;
                var bounds_cond = x == 0 | x == size_x;
                
                using (Variable.If(x == 0)) 
                {
                    solution[t][x] = 0;
                }
                using (Variable.If(x > 0)) 
                {
                    solution[t][x] = -solution[t-2][x] + 4 * solution[t-1][x] - solution[t-1][x - 1] - solution[t-1][x + 1] + function_mean[x] * dx * dx;
                }
            }  
        }

        for (int t = 1; t < size_x; t++)
        {
            using (ForEachBlock x_block = Variable.ForEach(grid_x))
            {
                var x = x_block.Index;
                var bounds_cond = x == 0 | x == size_x;

                using (Variable.If(bounds_cond)) 
                {
                    function_mean[x] = 0;
                }
                using (Variable.IfNot(bounds_cond)) 
                {
                    function_mean[x] = (solution[t-1][x] + solution[t + 1][x] - 4 * solution[t][x] + solution[t][x - 1] + solution[t][x + 1]) / (dx * dx);
                }
            }
        }
        
        var t = t_block.Index;
            using (Variable.If(t == 0))
            {
                using (ForEachBlock x = Variable.ForEach(grid_x))
                {  
                    solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, 0.01); 
                }
            }
            using (Variable.If(t == 1))
            {
                using (ForEachBlock x_block = Variable.ForEach(grid_x))
                {
                    var x = x_block.Index;
                    using (Variable.If(x == 0)) 
                    {
                        solution[t][x] = 0;
                    }
                    using (Variable.If(x > 0))
                    {
                        solution[t][x] = (-solution[t - 1][x - 1] - solution[t - 1][x + 1] + 4 * solution[t - 1][x] + 2 * dx * derivative_start_function[x] + function_mean[x] * dx) / 2;
                    }
                }
            }
            using (Variable.If(t > 1))
            {
                using (ForEachBlock x_block = Variable.ForEach(grid_x))
                {
                    var x = x_block.Index;
                    using (Variable.If(x == 0)) 
                    {
                        //Variable.ConstrainEqual(solution[t][x], 0);
                        solution[t][x] = 0;
                    }
                    using (Variable.If(x > 0))
                    {
                        //Variable.ConstrainEqual(solution[t-1][x] + solution[t + 1][x] - 4 * solution[t][x] + solution[t][x - 1] + solution[t][x + 1] - function_mean[x] * dx * dx, 0);
                        solution[t][x] = -solution[t-2][x] + 4 * solution[t-1][x] - solution[t-1][x - 1] - solution[t-1][x + 1] + function_mean[x] * dx * dx;
                    }
                }
            }*/

        using (ForEachBlock t_block = Variable.ForEach(grid_tt))
        {
            var t = t_block.Index;
            using (ForEachBlock x_block = Variable.ForEach(grid_xx))
            {
                var x = x_block.Index;
                using (Variable.If(t > 0))
                {
                    using (Variable.If(x == 0)) 
                    {
                        Variable.ConstrainEqual(solution[t][x], 0);
                    }
                    using (Variable.If(x > 0)) 
                    { 
                        Variable.ConstrainEqual(solution[t-1][x] + solution[t + 1][x] - 4 * solution[t][x] + solution[t][x - 1] + solution[t][x + 1] - (c1 * norm_dist_var[x + 130] + c2 * norm_dist_var[x + 100] + c3 * norm_dist_var[x + 70]) * function_t[t] * dx * dx, 0);
                    }
                    Variable.ConstrainEqual(solution[t][size_x], 0);
                }
            }
        }
        
        using (ForEachBlock x_block = Variable.ForEach(grid_xx))
        {
            var x = x_block.Index;
            using (Variable.If(x > 0))
            {
                //derivative_start_function[x] = Variable.GaussianFromMeanAndVariance((-solution[1][x] - 0.5 * solution[0][x - 1] - 0.5 * solution[0][x + 1] + 2 * solution[0][x]) / dx, 0.0001);
                //derivative_end_function[x] = Variable.GaussianFromMeanAndVariance((-solution[size_x - 1][x] - 0.5 * solution[size_x][x - 1] - 0.5 * solution[size_x][x + 1] + 2 * solution[size_x][x]) / dx, 0.01);
                Variable.ConstrainEqual(2 * solution[1][x] + solution[0][x - 1] + solution[0][x + 1] - 4 * solution[0][x] - 2 * dx * derivative_start_function[x], 0);
                Variable.ConstrainEqual(2 * solution[size_x - 1][x] + solution[size_x][x - 1] + solution[size_x][x + 1] - 4 * solution[size_x][x] - 2 * dx * derivative_end_function[x], 0);
            }
        }
        
        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            end_function[x.Index] = solution[size_x][x.Index];
            start_function[x.Index] = solution[0][x.Index];
        }
       
        InferenceEngine engine = new InferenceEngine();
        engine.SaveFactorGraphToFolder = "graphs";
        engine.Compiler.UseParallelForLoops = true;
        engine.NumberOfIterations = 500;

        //Console.WriteLine(engine.Infer(solution));
        var prediction_c1 = engine.Infer<Gaussian>(c1);
        var prediction_c2 = engine.Infer<Gaussian>(c2);
        var prediction_c3 = engine.Infer<Gaussian>(c3);
        Console.WriteLine(prediction_c1);
        //Console.WriteLine(engine.Infer(c1_prec));
        Console.WriteLine(prediction_c2);
        //Console.WriteLine(engine.Infer(c2_prec));
        Console.WriteLine(prediction_c3);
        //Console.WriteLine(engine.Infer(c3_prec));
        
        //Console.WriteLine(prediction_loc);
        var result = new double[size_x + 1];
        for (int i = 0; i <= size_x; i++)
        {
            result[i] = prediction_c1.GetMean() * norm_dist(i * dx + 0.3) + prediction_c2.GetMean() * norm_dist(i * dx) + prediction_c3.GetMean() * norm_dist(i * dx - 0.3);
        }
        

        using (StreamWriter writer = new StreamWriter(args[6]))
        {
            foreach (double d in result)
                writer.WriteLine(d.ToString("F18", culture_param));
        }
    }
}


