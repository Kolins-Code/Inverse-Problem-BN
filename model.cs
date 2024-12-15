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


class Model
{
    static void Main(string[] args)
    {
        //args = new[]{"0.0001", "0.01", "0.0002", "1", "D:/Programming/InverseProblemBN/Runs/run 19.11.24 - 11.27.49/input/data.csv", "D:/Programming/InverseProblemBN/Runs/run 19.11.24 - 11.27.49/output/data.csv"};
        var culture_param = new CultureInfo("en-EN");

        double dt = Convert.ToDouble(args[0], culture_param);
        double dx = Convert.ToDouble(args[1], culture_param);
        double end_x = Convert.ToDouble(args[3], culture_param);
        double end_t = Convert.ToDouble(args[2], culture_param);

        int size_x = (int) (end_x / dx);
        int size_t_big = (int) (end_t / dt);

        int size_t = size_t_big;

        double gamma = dt / (dx * dx);

        var reader = new StreamReader(args[4]);

        double[] end_function_obs = new double[size_x + 1];

        int k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            end_function_obs[k] = Convert.ToDouble(line, culture_param);
            k++;
        }
        reader.Close();


        Range grid_x = new Range(size_x + 1).Named("Сетка по x");
        Range grid_t = new Range(size_t + 1).Named("Сетка по t");
        
        //VariableArray<double>[] solution = new VariableArray<double>[size_t + 1];
        VariableArray2D<double> precisions_eq = Variable.Array<double>(grid_t, grid_x);
        precisions_eq[grid_t, grid_x] = Variable.GammaFromMeanAndVariance(1, 1).ForEach(grid_t, grid_x);
        //VariableArray2D<double> solution = Variable.Array<double>(grid_t, grid_x);
        VariableArray<VariableArray<double>,double[][]> solution = Variable.Array(Variable.Array<double>(grid_x), grid_t).Named("Решение прямой задачи");
        VariableArray<double> start_function = Variable.Array<double>(grid_x).Named("Начальное условие");
        VariableArray<double> end_function = Variable.Observed(end_function_obs);

        //VariableArray<double> means = Variable.Array<double>(grid_x);
        //VariableArray<double> precisions = Variable.Array<double>(grid_t);
        //VariableArray<double> precisions = Variable.Array<double>(grid_t);
        Variable<double> means_precision = Variable.GammaFromMeanAndVariance(0.1, 0.1);
        //means[grid_x] = Variable.GaussianFromMeanAndVariance(0, 1000).ForEach(grid_x);
        //means[grid_x] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(grid_x);
        //Variable<double> precision_of_precision =  Variable.GammaFromMeanAndVariance(1000, 100);
        //Variable<double> shape =  Variable.GammaFromMeanAndVariance(10, 0.1);
        VariableArray<double> precisions = Variable.Array<double>(grid_t);
        precisions[grid_t] = Variable.GammaFromMeanAndVariance(1, 1).ForEach(grid_t);
        VariableArray<double>[] means = new VariableArray<double>[size_t + 1];
        //means[grid_t, grid_x] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(grid_t, grid_x);
        var precision =  Variable.GammaFromMeanAndVariance(1, 1);
        //precision.ObservedValue = 10;
        VariableArray<double>[]  addition = new VariableArray<double>[size_t + 1];
        VariableArray<double>[]  addition_means = new VariableArray<double>[size_t + 1];
        VariableArray<double>[]  addition_prec = new VariableArray<double>[size_t + 1];
        //addition[grid_t, grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_t, grid_x);
        //Variable<double> m = Variable.GaussianFromMeanAndVariance(2, 0.1);
        //Variable.ConstrainTrue(m >= 1);

        /*for (int i = 0; i < solution.Length; i++)
        {
            solution[i] = Variable.Array<double>(grid_x).Named("Решение прямой задачи, слой " + i);
            means[i] = Variable.Array<double>(grid_x);
        }
        for (int i = 0; i < solution.Length; i++)
        {
            means[i][grid_x] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(grid_x);
            
            /*for (int j = 0; j <= size_x; j++)
            { 
                means[i][j] = Variable.GaussianFromMeanAndPrecision(/*Rand.Double()0, 0.01);
            }
            
        }*/
        /*for (int i = 0; i < addition.Length; i++)
        {
            addition[i] = Variable.Array<double>(grid_x);
            addition_means[i] = Variable.Array<double>(grid_x);
            addition_means[i][grid_x] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_x);
            addition_prec[i] = Variable.Array<double>(grid_x);
            addition_prec[i][grid_x] = Variable.GammaFromMeanAndVariance(1, 0.001).ForEach(grid_x);
            addition[i][grid_x] = Variable.GaussianFromMeanAndPrecision(addition_means[i][grid_x], addition_prec[i][grid_x]);
        }*/

        /*for (int i = 1; i < size_x; i++)
        {
            //solution[0][i] = Variable.GaussianFromMeanAndPrecision(0,  0.01);
            solution[0][i] = Variable.GaussianFromMeanAndPrecision(means[0][i], precisions[0]);
        }
        solution[0][0] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
        solution[0][size_x] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);*/

        /*using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            using (ForEachBlock t = Variable.ForEach(grid_t))
            {
                var left_cond = x.Index == 0 | x.Index == size_x;
                //var right_cond = x.Index == size_x;
                /*var start_cond = t.Index == 0;
                var end_cond = t.Index == size_t;
                var t_bounds = start_cond | end_cond;*/
                //solution[t.Index, x.Index] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
                /*using (Variable.If(left_cond)) 
                {
                    solution[t.Index][x.Index] = Variable.GaussianFromMeanAndVariance(0, 0.00001);
                    /*using (Variable.IfNot(end_cond)) 
                    {
                        solution[t.Index][x.Index] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
                    }*/
                /*}
                using (Variable.IfNot(left_cond)) 
                {
                    solution[t.Index][x.Index] = Variable.GaussianFromMeanAndPrecision(means[t.Index], precisions[t.Index]);
                    
                }
            }  
        }*/

        /*using (ForEachBlock t = Variable.ForEach(grid_t))
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {
                var left_cond = x.Index == 0 | x.Index == size_x;
                var start_cond = t.Index == 0;
                using (Variable.If(left_cond)) 
                {
                    //Variable.ConstrainEqual(solution[t.Index, x.Index], 0);
                    solution[t.Index, x.Index] = Variable.GaussianFromMeanAndVariance(0, 0);
                }
                using (Variable.IfNot(left_cond)) 
                {
                    using (Variable.If(start_cond)) 
                    {
                        solution[t.Index, x.Index] = Variable.GaussianFromMeanAndPrecision(means[x.Index], precisions[t.Index]);
                    }
                    using (Variable.IfNot(start_cond)) 
                    {
                        /*var tmp1 = (solution[t.Index - 1, x.Index - 1] + solution[t.Index - 1, x.Index + 1]).Named("tmp1");
                        var tmp2 = (gamma * tmp1).Named("tmp2");
                        var tmp3 = (solution[t.Index - 1, x.Index] * (1 + 2 * gamma)).Named("tmp3");*/
                        //solution[t.Index][x.Index] = tmp3 / (1 + 2 * gamma);
                        //Variable.ConstrainEqual(solution[t.Index][x.Index - 1], tmp3 / (1 + 2 * gamma));
                        /*var tmp1 = (solution[t.Index - 1, x.Index - 1] + solution[t.Index - 1, x.Index + 1]).Named("tmp1");
                        var tmp2 = (gamma * tmp1).Named("tmp2");
                        var tmp3 = ((1 - 2 * gamma) * solution[t.Index - 1, x.Index]).Named("tmp3");*/
                        //Variable.ConstrainEqual(solution[t.Index, x.Index], tmp2 + tmp3);
                        //solution[t.Index, x.Index] = tmp2 + tmp3;
                        //var mean = Variable.GaussianFromMeanAndPrecision(0, 0.01);
                        //Variable.ConstrainEqual(mean, tmp2 + tmp3);
                        //solution[t.Index, x.Index] = Variable.GaussianFromMeanAndPrecision(tmp3 + tmp2, precisions[t.Index]);
                        //Variable.ConstrainBetween(solution[t.Index, x.Index], 0, 5);
                        //solution[t.Index][x.Index] = (solution[t.Index - 1][x.Index] + gamma * (solution[t.Index][x.Index - 1] + solution[t.Index][x.Index + 1])) / (1 + 2 * gamma) ;
                        //solution[t.Index][x.Index] = ((1 + 2 * gamma) * solution[t.Index][x.Index - 1] - solution[t.Index - 1][x.Index - 1] - gamma * solution[t.Index][x.Index - 2]) / gamma;
                        /*var tmp1 = ((1 + 2 * gamma) * solution[t.Index][x.Index - 1]).Named("tmp1");
                        var tmp2 = (tmp1 - - solution[t.Index - 1][x.Index - 1]).Named("tmp2");
                        var tmp3 = (tmp2 - gamma * solution[t.Index][x.Index - 2]).Named("tmp3");
                        solution[t.Index][x.Index] = tmp3 / gamma;
                    }
                    
                    
                }
            }
        }*/

        for (int t = size_t; t >= 0; t--)
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {  
                solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, 0.01); 
            }  
        }
        //solution[size_t][grid_x] = Variable.GaussianFromMeanAndPrecision(means[size_t][grid_x], 1000); 
        for (int t = 0; t < size_t; t++)
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {
                var bounds_cond = x.Index == 0 | x.Index == size_x;
                using (Variable.If(bounds_cond)) 
                {
                    Variable.ConstrainEqualRandom(solution[t][x.Index], Gaussian.FromMeanAndVariance(0, 0.01));
                    //Variable.ConstrainEqual(means[t][x.Index], 0);
                    //solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
                    //solution_non_norm[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
                }
                using (Variable.IfNot(bounds_cond)) 
                {
                    //solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(means[t][x.Index], precisions[t, x.Index]);
                    //Variable.ConstrainEqual(solution[t][x.Index], (1 + 2 * gamma) * means[t + 1][x.Index] - gamma * (means[t + 1][x.Index - 1] + means[t + 1][x.Index + 1]));
                    //var cond = Variable.GaussianFromMeanAndPrecision(0, 100);
                    /*var sign = solution[t][x.Index] > 0;
                    Variable<double> abs = Variable.New<double>(); 
                    using (Variable.If(sign)) 
                    {
                        abs.SetTo(solution[t][x.Index]);
                    }
                    using (Variable.IfNot(sign)) 
                    {
                        abs.SetTo(-solution[t][x.Index]);
                    }*/
                    Variable.ConstrainEqualRandom(solution[t][x.Index] - (1 + 2 * gamma) * solution[t + 1][x.Index] + gamma * (solution[t + 1][x.Index - 1] + solution[t + 1][x.Index + 1]), Gaussian.FromMeanAndVariance(0, 0.01));
                    //Variable.ConstrainEqual(solution[t][x.Index], ((1 + 2 * gamma) * means[t + 1][x.Index] - gamma * (means[t + 1][x.Index - 1] + means[t + 1][x.Index + 1]) + 0.4 * abs) / (1 + 0.4));
                    //solution[t][x.Index] = (1 + 2 * gamma) * Variable.GaussianFromMeanAndPrecision(solution[t + 1][x.Index], precision) - gamma * (Variable.GaussianFromMeanAndPrecision(solution[t + 1][x.Index - 1], precision) + Variable.GaussianFromMeanAndPrecision(solution[t + 1][x.Index + 1], precision));
                    
                    //solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision((1 - 2 * gamma) * solution[t - 1][x.Index] + gamma * (solution[t - 1][x.Index - 1] + solution[t - 1][x.Index + 1]), precisions[t]);
                    //solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision((solution_non_norm[t][x.Index - 1] + solution_non_norm[t][x.Index] + solution_non_norm[t][x.Index + 1]) / 3, 100);
                }
            }  
        }

        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            end_function[x.Index] = solution[size_t][x.Index];
        }
        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            start_function[x.Index] = solution[0][x.Index];
        }
        /*solution[size_t].ObservedValue = end_function_obs;

        InferenceEngine engine = new InferenceEngine();
        engine.SaveFactorGraphToFolder = "graphs";
        engine.NumberOfIterations = 10;*/
        //engine.Compiler.BrowserMode = BrowserMode.Always;

        
        //var prediction_mean = engine.Infer(means);
        
        //var prediction_prec_prec = engine.Infer(precision_of_precision);
        //var prediction_m = engine.Infer(m);
        //var means_precision_prec = engine.Infer(means_precision);
        //var prediction_prec = engine.Infer(precisions);
        
        //Console.WriteLine(prediction_mean);
        //Console.WriteLine(prediction_m);
        //Console.WriteLine(means_precision_prec);
        
        //Console.WriteLine(prediction_prec_prec);
        //Console.WriteLine(engine.Infer(addition_prec[size_t]));
        InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
        engine.SaveFactorGraphToFolder = "graphs";
        engine.Compiler.UseParallelForLoops = true;
        engine.NumberOfIterations = 500000;

        for (int i = 0; i < size_t_big / size_t; i++)
        {
            end_function.ObservedValue = end_function_obs;
            var prediction_loc = engine.Infer<DistributionStructArray<Gaussian, double>>(start_function);
            Console.WriteLine(engine.Infer(precisions));
            //Console.WriteLine(engine.Infer(precisions_eq));
            Console.WriteLine(prediction_loc);
            int index_loc = 0;
            var result = new double[size_x + 1];
            foreach (var point in prediction_loc)
            {
                result[index_loc++] = point.GetMean();
            }
            for (int j = 1; j < size_x; j++)
            {
                end_function_obs[j] = /*(result[j - 1] + result[j] + result[j + 1]) / 3;*/result[j];
            }
        }
        //Console.WriteLine(prediction_prec);

        /*var prediction = engine.Infer<DistributionStructArray<Gaussian, double>>(means[0]);
        Console.WriteLine(prediction);
        double[] output_function = new double[size_x + 1];
        int index = 0;
        foreach (var point in prediction)
        {
            output_function[index++] = point.GetMean();
        }*/

        using (StreamWriter writer = new StreamWriter(args[5]))
        {
            foreach (double d in end_function_obs)
                writer.WriteLine(d.ToString("F18", culture_param));
        }
    }
}


