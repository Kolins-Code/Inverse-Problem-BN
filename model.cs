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
        
        VariableArray2D<double> precisions_eq = Variable.Array<double>(grid_t, grid_x);
        precisions_eq[grid_t, grid_x] = Variable.GammaFromMeanAndVariance(1, 1).ForEach(grid_t, grid_x);
        
        VariableArray<VariableArray<double>,double[][]> solution = Variable.Array(Variable.Array<double>(grid_x), grid_t).Named("Решение прямой задачи");
        VariableArray<double> start_function = Variable.Array<double>(grid_x).Named("Начальное условие");
        VariableArray<double> end_function = Variable.Observed(end_function_obs);

        Variable<double> means_precision = Variable.GammaFromMeanAndVariance(0.1, 0.1);
        VariableArray<double> precisions = Variable.Array<double>(grid_t);
        precisions[grid_t] = Variable.GammaFromMeanAndVariance(1, 1).ForEach(grid_t);
        VariableArray<double>[] means = new VariableArray<double>[size_t + 1];
       
        var precision =  Variable.GammaFromMeanAndVariance(1, 1);
      
        VariableArray<double>[]  addition = new VariableArray<double>[size_t + 1];
        VariableArray<double>[]  addition_means = new VariableArray<double>[size_t + 1];
        VariableArray<double>[]  addition_prec = new VariableArray<double>[size_t + 1];

        for (int t = size_t; t >= 0; t--)
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {  
                solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, 0.01); 
            }  
        }
       
        for (int t = 0; t < size_t; t++)
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {
                var bounds_cond = x.Index == 0 | x.Index == size_x;
                using (Variable.If(bounds_cond)) 
                {
                    Variable.ConstrainEqualRandom(solution[t][x.Index], Gaussian.FromMeanAndVariance(0, 0.01));
                }
                using (Variable.IfNot(bounds_cond)) 
                {
                    
                    Variable.ConstrainEqualRandom(solution[t][x.Index] - (1 + 2 * gamma) * solution[t + 1][x.Index] + gamma * (solution[t + 1][x.Index - 1] + solution[t + 1][x.Index + 1]), Gaussian.FromMeanAndVariance(0, 0.01));
                    
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
       
        InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
        engine.SaveFactorGraphToFolder = "graphs";
        engine.Compiler.UseParallelForLoops = true;
        engine.NumberOfIterations = 100000;

        for (int i = 0; i < size_t_big / size_t; i++)
        {
            end_function.ObservedValue = end_function_obs;
            var prediction_loc = engine.Infer<DistributionStructArray<Gaussian, double>>(start_function);
            Console.WriteLine(engine.Infer(precisions));
           
            Console.WriteLine(prediction_loc);
            int index_loc = 0;
            var result = new double[size_x + 1];
            foreach (var point in prediction_loc)
            {
                result[index_loc++] = point.GetMean();
            }
            for (int j = 1; j < size_x; j++)
            {
                end_function_obs[j] = result[j];
            }
        }

        using (StreamWriter writer = new StreamWriter(args[5]))
        {
            foreach (double d in end_function_obs)
                writer.WriteLine(d.ToString("F18", culture_param));
        }
    }
}


