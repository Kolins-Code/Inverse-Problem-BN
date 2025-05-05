using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

using System.Globalization;

class Model2
{
    static void Main(string[] args)
    {
        //args = new[]{"0.00025", "0.01", "0.005", "1", "D:/Programming/InverseProblemBN/Runs/run 12.03.25 - 19.20.21/input/data.csv", "D:/Programming/InverseProblemBN/Runs/run 12.03.25 - 19.20.21/output/data.csv"};
        var culture_param = new CultureInfo("en-EN");

        double dt = Convert.ToDouble(args[0], culture_param);
        double dx = Convert.ToDouble(args[1], culture_param);
        double end_x = Convert.ToDouble(args[3], culture_param);
        double end_t = Convert.ToDouble(args[2], culture_param);

        int size_x = (int) (end_x / dx);
        int size_t_big = (int) (end_t / dt);

        int size_t = size_t_big;

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

        int size_coef = 25;

        Range grid_x = new Range(size_x + 1).Named("Сетка по x");
        Range grid_t = new Range(size_t + 1).Named("Сетка по t");

        Range grid_coef = new Range(size_coef);

        VariableArray<double> end_function = Variable.Array<double>(grid_x);

        VariableArray<double> coefficients = Variable.Array<double>(grid_coef);
        VariableArray<double> coefficients_mean = Variable.Array<double>(grid_coef);
        VariableArray<double> coefficients_prec = Variable.Array<double>(grid_coef);
        //coefficients_mean[grid_coef] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(grid_coef);
        //coefficients_prec[grid_coef] = Variable.GammaFromMeanAndVariance(1, 1).ForEach(grid_coef);
        coefficients[grid_coef] = Variable.GaussianFromMeanAndPrecision(0/*coefficients_mean[grid_coef]*/, 0.01/*coefficients_prec[grid_coef]*/).ForEach(grid_coef);

        double[,] sines_obs = new double[size_x + 1, size_coef];
        double[] exps_obs = new double[size_coef];
        for (int i = 0; i < size_x + 1; i++)
        {
            for (int j = 0; j < size_coef; j++)
            {
                sines_obs[i, j] = Math.Sin(Math.PI * (j + 1) * i * dx / end_x);
            }
        }
        for (int j = 0; j < size_coef; j++)
        {
            exps_obs[j] = Math.Exp(-(Math.PI * (j + 1) / end_x) * (Math.PI * (j + 1) / end_x) * end_t);
        }
        VariableArray2D<double> sines = Variable.Array<double>(grid_x, grid_coef);
        sines.ObservedValue = sines_obs;
        VariableArray<double> exps = Variable.Array<double>(grid_coef);
        exps.ObservedValue = exps_obs;
        

        using (ForEachBlock x = Variable.ForEach(grid_x))
        {
            VariableArray<double> sums = Variable.Array<double>(grid_coef);
            using (ForEachBlock c = Variable.ForEach(grid_coef))
            {
                sums[c.Index] = coefficients[c.Index] * sines[x.Index, c.Index] * exps[c.Index];
            }
            Variable.ConstrainEqualRandom(end_function[x.Index] - Variable.Sum(sums), Gaussian.FromMeanAndVariance(0, 1e-2));
        }

        /*for (int i = 0; i <= size_x; i++)
        {
            VariableArray<double> sums = Variable.Array<double>(grid_coef);
            for (int j = 0; j < size_coef; j++)
            {
                sums[j] = coefficients[j] * Math.Sin(Math.PI * (j + 1) * i * dx / end_x) * Math.Exp(-(Math.PI * (j + 1) / end_x) * (Math.PI * (j + 1) / end_x) * end_t);
            }
            Variable.ConstrainEqualRandom(end_function[i] - Variable.Sum(sums), Gaussian.FromMeanAndVariance(0, 0.01));
        }*/

        InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
        engine.SaveFactorGraphToFolder = "graphs";
        engine.Compiler.UseParallelForLoops = true;
        engine.NumberOfIterations = 100;

        end_function.ObservedValue = end_function_obs;
        
        var prediction = engine.Infer<DistributionStructArray<Gaussian, double>>(coefficients);
        Console.WriteLine(prediction);
        //Console.WriteLine(engine.Infer(coefficients_prec));
        for (int i = 1; i < size_x; i++)
        {
            end_function_obs[i] = 0;
            int j = 1;
            foreach (var coef in prediction)
            {
                end_function_obs[i] += coef.GetMean() * Math.Sin(Math.PI * j * i * dx / end_x);
                j++;
            }
        }

        using (StreamWriter writer = new StreamWriter(args[5]))
        {
            foreach (double d in end_function_obs)
                writer.WriteLine(d.ToString("F18", culture_param));
        }
    }
}