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
        int size_t = (int) (end_t / dt);

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
        
        VariableArray<VariableArray<double>,double[][]> solution = Variable.Array(Variable.Array<double>(grid_x), grid_t).Named("Решение прямой задачи");
        VariableArray<double> start_function = Variable.Array<double>(grid_x).Named("Начальное условие");
        VariableArray<double> end_function = Variable.Observed(end_function_obs);


        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            using (ForEachBlock t = Variable.ForEach(grid_t))
            {
                var left_cond = x.Index == 0 | x.Index == size_x;
                //var right_cond = x.Index == size_x;
                var start_cond = t.Index == 0;
                var end_cond = t.Index == size_t;
                var t_bounds = start_cond | end_cond;

                using (Variable.If(left_cond)) 
                {
                    //solution[t.Index][x.Index] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
                    using (Variable.IfNot(end_cond)) 
                    {
                        solution[t.Index][x.Index] = Variable.GaussianFromMeanAndPrecision(0, Double.PositiveInfinity);
                    }
                }
                using (Variable.IfNot(left_cond)) 
                {
                    using (Variable.If(start_cond)) 
                    {
                        solution[t.Index][x.Index] = Variable.GaussianFromMeanAndVariance(0, 100);
                      
                    }
                    using (Variable.IfNot(start_cond)) 
                    {
                        solution[t.Index][x.Index] = Variable.GaussianFromMeanAndVariance(0, 100);
                        var tmp1 = (solution[t.Index][x.Index - 1] + solution[t.Index][x.Index + 1]).Named("tmp1");
                        var tmp2 = (gamma * tmp1).Named("tmp2");
                        var tmp3 = (solution[t.Index - 1][x.Index] + tmp2).Named("tmp3");
                        //solution[t.Index][x.Index] = tmp3 / (1 + 2 * gamma);
                        Variable.ConstrainEqual(solution[t.Index][x.Index], tmp3 / (1 + 2 * gamma));
                        //solution[t.Index][x.Index] = (solution[t.Index - 1][x.Index] + gamma * (solution[t.Index][x.Index - 1] + solution[t.Index][x.Index + 1])) / (1 + 2 * gamma) ;
                        //solution[t.Index][x.Index] = ((1 + 2 * gamma) * solution[t.Index][x.Index - 1] - solution[t.Index - 1][x.Index - 1] - gamma * solution[t.Index][x.Index - 2]) / gamma;
                        /*var tmp1 = ((1 + 2 * gamma) * solution[t.Index][x.Index - 1]).Named("tmp1");
                        var tmp2 = (tmp1 - - solution[t.Index - 1][x.Index - 1]).Named("tmp2");
                        var tmp3 = (tmp2 - gamma * solution[t.Index][x.Index - 2]).Named("tmp3");
                        solution[t.Index][x.Index] = tmp3 / gamma;*/
                    }
                    
                    
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


        InferenceEngine engine = new InferenceEngine();
        engine.SaveFactorGraphToFolder = "graphs";
        engine.NumberOfIterations = 50;
        //engine.Compiler.BrowserMode = BrowserMode.Always;

        var prediction = engine.Infer<DistributionStructArray<Gaussian, double>>(start_function);
        Console.WriteLine(prediction);
        double[] output_function = new double[size_x + 1];
        int index = 0;
        foreach (var point in prediction)
        {
            output_function[index++] = point.GetMean();
        }

        using (StreamWriter writer = new StreamWriter(args[5]))
        {
            foreach (double d in output_function)
                writer.WriteLine(d.ToString("F18", culture_param));
        }
    }
}


