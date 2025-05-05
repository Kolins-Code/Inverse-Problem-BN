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


class ModelLaplace
{
    static void Main(string[] args)
    {
        //args = new[]{"0.0001", "0.01", "D:/Programming/InverseProblemBN/Runs/run 19.11.24 - 11.27.49/input/data.csv", "D:/Programming/InverseProblemBN/Runs/run 19.11.24 - 11.27.49/output/data.csv"};
        var culture_param = new CultureInfo("en-EN");
       
        double dx = Convert.ToDouble(args[0], culture_param);
        double end_x = Convert.ToDouble(args[1], culture_param);

        int size_x = (int) (end_x / dx);

        var reader = new StreamReader(args[2]);

        double[] function_obs = new double[size_x + 1];

        int k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            function_obs[k] = Convert.ToDouble(line, culture_param);
            k++;
        }
        reader.Close();
        reader = new StreamReader(args[3]);

        double[] derivative_function_obs = new double[size_x + 1];

        k = 0;
        while (!reader.EndOfStream && k < size_x + 1)
        {
            var line = reader.ReadLine();
            derivative_function_obs[k] = Convert.ToDouble(line, culture_param);
            k++;
        }
        reader.Close();


        Range grid_x = new Range(size_x + 1).Named("Сетка по x");
        Range grid_t = grid_x.Clone();
        
        VariableArray<VariableArray<double>,double[][]> solution = Variable.Array(Variable.Array<double>(grid_x), grid_t).Named("Решение прямой задачи");
        VariableArray<double> end_function = Variable.Array<double>(grid_x);
        VariableArray<double> end_function_mean = Variable.Array<double>(grid_x);
        VariableArray<double> end_function_prec = Variable.Array<double>(grid_x);
        VariableArray<double> start_function = Variable.Observed(function_obs);
        VariableArray<double> derivative_function = /*Variable.Array<double>(grid_x)*/Variable.Observed(derivative_function_obs);

        //Variable<double> prec = Variable.GammaFromMeanAndVariance(100, 10);

        /*for (int x = 0; x <= size_x; x++)
        {
            derivative_function[x] = Variable.GaussianFromMeanAndVariance(derivative_function_obs[x], 0.0001);
        }*/

        for (int t = size_x; t >= 0; t--)
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {  
                solution[t][x.Index] = Variable.GaussianFromMeanAndPrecision(0, 0.01); 
            }  
        }
        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            end_function_mean[x.Index] = Variable.GaussianFromMeanAndVariance(0, 100);
            end_function_prec[x.Index] = Variable.GammaFromShapeAndScale(1, 1);
        }
         
        /*for (int x = 1; x < size_x; x++){
            //solution[size_x][x] = Variable.GaussianFromMeanAndPrecision(end_function_mean[x], end_function_prec[x]);
            //solution[size_x - 1][x] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
            solution[0][x] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
        }*/
       
        /*for (int t = 0; t < size_x - 1; t++)
        {
            using (ForEachBlock x = Variable.ForEach(grid_x))
            {
                var bounds_cond = x.Index == 0 | x.Index == size_x;
                using (Variable.If(x.Index == 0)) 
                {
                    Variable.ConstrainEqualRandom(solution[t][x.Index], Gaussian.FromMeanAndVariance(0, 0.0001));
                }
                /*using (Variable.If(x.Index == size_x)) 
                {
                    Variable.ConstrainEqualRandom(solution[t][x.Index], Gaussian.FromMeanAndVariance(0, 0.0001));
                }
                using (Variable.If(x.Index > 0))
                { 
                    var sum = solution[t + 1][x.Index - 1] + solution[t + 1][x.Index];
                    Variable.ConstrainEqualRandom(solution[t][x.Index] + sum, Gaussian.FromMeanAndVariance(0, 0.01));
                }
            }  
        }*/
        
        for (int t = 1; t < size_x; t++)
        {
            for (int x = 0; x <= size_x; x++)
            {
                if (x == 0 || x == size_x){
                    Variable.ConstrainEqualRandom(solution[t][x], Gaussian.FromMeanAndVariance(0, 0.00001));
                } else {
                    
                    Variable.ConstrainEqualRandom(solution[t-1][x] + solution[t + 1][x] - 4 * solution[t][x] + solution[t][x - 1] + solution[t][x + 1], Gaussian.FromMeanAndVariance(0, 0.00001));
                }
            }  
        }
        /*solution[0][0] = 0;
        solution[0][size_x] = 0;
        solution[1][0] = 0;
        solution[1][size_x] = 0;*/
        /*for (int x = 1; x < size_x; x++)
        {
            solution[1][x] =  -0.5 * solution[0][x - 1] - 0.5 * solution[0][x + 1] + 2 * solution[0][x] + dx * Variable.GaussianFromMeanAndPrecision(derivative_function[x], end_function_prec[x]);
            //derivative_function[x] = Variable.GaussianFromMeanAndVariance(solution[1][x] / dx, 0.01);
        }*/
        
        /*for (int t = 2; t <= size_x; t++)
        {
            for (int x = 0; x <= size_x; x++)
            {
                if (x == 0 || x == size_x){
                    solution[t][x] = 0;
                } else {
                    solution[t][x] = -solution[t - 2][x] + 4 * solution[t - 1][x] - solution[t - 1][x - 1] - solution[t - 1][x + 1];
                }
            }
        }*/
        /*using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            var bounds_cond = x.Index == 0 | x.Index == size_x;
            using (Variable.IfNot(bounds_cond)) 
            { 
                Variable.ConstrainEqualRandom(2 * solution[1][x.Index] + solution[0][x.Index - 1] + solution[0][x.Index + 1] - 4 * solution[0][x.Index] - 2 * dx * derivative_function[x.Index], Gaussian.FromMeanAndVariance(0, 1e-10));
            }
        }*/
        for (int x = 1; x < size_x; x++)
        {
            Variable.ConstrainEqualRandom(2 * solution[1][x] + solution[0][x - 1] + solution[0][x + 1] - 4 * solution[0][x] - 2 * dx * derivative_function[x], Gaussian.FromMeanAndVariance(0, 0.00001));
        }
        for (int x = 1; x < size_x; x++)
        {
            //solution[1][x] = -0.5 * solution[0][x - 1] - 0.5 * solution[0][x + 1] + 2 * solution[0][x] + dx * Variable.GaussianFromMeanAndVariance(derivative_function[x], 0.01);
            //derivative_function[x] = solution[1][x] / dx;
        }
        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            end_function[x.Index] = solution[size_x][x.Index];
            //Variable.ConstrainEqual(derivative_function[x.Index] * dx, solution[1][x.Index]);
            //end_function[x.Index] = Variable.GaussianFromMeanAndPrecision(end_function_mean[x.Index], end_function_prec[x.Index]);
            //Variable.ConstrainEqualRandom(end_function[x.Index] - solution[size_x][x.Index], Gaussian.FromMeanAndVariance(0, 0.01));
        }
        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            start_function[x.Index] = solution[0][x.Index];
        }
       
        InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
        engine.SaveFactorGraphToFolder = "graphs";
        engine.Compiler.UseParallelForLoops = true;
        engine.NumberOfIterations = 10000000;

        //Console.WriteLine(engine.Infer(solution));
        var prediction_loc = engine.Infer<DistributionStructArray<Gaussian, double>>(end_function);
        Console.WriteLine(engine.Infer(end_function_prec));
        
        Console.WriteLine(prediction_loc);
        int index_loc = 0;
        var result = new double[size_x + 1];
        foreach (var point in prediction_loc)
        {
            result[index_loc++] = point.GetMean();
        }
        for (int j = 1; j < size_x; j++)
        {
            function_obs[j] = result[j];
        }
        

        using (StreamWriter writer = new StreamWriter(args[4]))
        {
            foreach (double d in function_obs)
                writer.WriteLine(d.ToString("F18", culture_param));
        }
    }
}


