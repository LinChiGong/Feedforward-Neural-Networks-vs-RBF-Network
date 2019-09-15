/**
 * This class performs classification using Feedforward Neural Network and RBF
 * Network on the specified input file and prints the processes and results to
 * the console. Note that this class serves a demonstration purpose. For a
 * complete run on all 5 datasets used in this project, run the "WriteToFile"
 * class
 * 
 * @author Winston Lin
 */

import java.io.*;
import java.util.*;

public class Main 
{
    public static void main(String[] args) throws IOException
    {
        String inputFile;
        
        // Prompt for the dataset to be used
        Scanner scan = new Scanner(System.in);
        System.out.println();
        System.out.print("Specify an input file: ");
        inputFile = scan.nextLine().trim();
        System.out.println();
        
        // Preprocess and split the data into training set and test test
        ArrayList<String[]> records = new ArrayList<String[]>();
        ArrayList<ArrayList<String[]>> partitions = 
                new ArrayList<ArrayList<String[]>>();
        
        ETL etl = new ETL();
        records = etl.readCSV(inputFile);
        partitions = etl.split(records, true);
        
        for (int a = 0; a < 4; a++)
        {
            if (a < 3)
            {
                // Checkpoint
                System.out.print("Press 'Enter' to run the Feedforward Neural "
                        + "Network:");
                String temp = scan.nextLine().trim();
                System.out.println();
            }
            else
            {
                System.out.print("Press 'Enter' to run the Radial Basis "
                        + "Function Network:");
                String temp = scan.nextLine().trim();
                System.out.println();
            }
            
            for (int k = 0; k < 5; k++)
            {
                ArrayList<String[]> Xtrain = new ArrayList<String[]>();
                ArrayList<String> ytrain = new ArrayList<String>();
                ArrayList<String[]> Xtest = new ArrayList<String[]>();
                ArrayList<String> ytest = new ArrayList<String>();
                
                for (int i = 0; i < partitions.size(); i++)
                {
                    if (i == k)
                    {
                        for (String[] test : partitions.get(i))
                        {
                            String[] X = new String[test.length - 1];
                            for (int j = 0; j < X.length; j++)
                            {
                                X[j] = test[j];
                            }
                            Xtest.add(X);
                            ytest.add(test[test.length - 1]);
                        }
                    }
                    else
                    {
                        for (String[] train : partitions.get(i))
                        {
                            String[] X = new String[train.length - 1];
                            for (int j = 0; j < X.length; j++)
                            {
                                X[j] = train[j];
                            }
                            Xtrain.add(X);
                            ytrain.add(train[train.length - 1]);
                        }
                    }
                }
                
                if (a < 3)
                {
                    System.out.println(" ------------------------- ");
                    System.out.println("| CROSS VALIDATION FOLD " + (k+1) 
                                                                  + " |");
                    System.out.println(" ------------------------- ");
                    
                    // Build and train the Feedforward Network model
                    ArrayList<String> ypred = new ArrayList<String>();
                    
                    FeedForwardNetwork ffn = 
                            new FeedForwardNetwork(a, -1, 0.95, 10000, 0.2);
                    ffn.fit(Xtrain, ytrain);
                    ypred = ffn.predict(Xtest);
                    
                    // Print the parameters and the prediction results
                    System.out.println();
                    System.out.println("Weights of each node");
                    System.out.println("--------------------");
                    
                    for (HiddenNode[] hiddenLayer : ffn.hiddenLayers)
                    {
                        System.out.println("Hidden layer: ");
                        int i = 1;
                        for (HiddenNode hNode : hiddenLayer)
                        {
                            System.out.print("Hidden node " + i + ": [ ");
                            for (double weight : hNode.weights)
                            {
                                System.out.print(weight + " ");
                            }
                            System.out.println("]");
                            i++;
                        }
                    }
                    System.out.println("Output layer: ");
                    int j = 1;
                    for (OutputNode oNode : ffn.outputLayer)
                    {
                        System.out.print("Output node " + j + ": [ ");
                        for (double weight : oNode.weights)
                        {
                            System.out.print(weight + " ");
                        }
                        System.out.println("]");
                        j++;
                    }
                    System.out.println();
                    
                    System.out.println("\tACTUAL" + "\t\tPREDICTED");
                    System.out.println("\t------" + "\t\t---------");
                    double performance = 0;
                    for (int i = 0; i < ypred.size(); i++)
                    {
                        System.out.println("\t" + ytest.get(i) + "\t| " 
                                                + ypred.get(i));
                        if (ytest.get(i).equals(ypred.get(i)))
                        {
                            performance++;
                        }
                    }
                    performance = performance / ytest.size();
                    System.out.println("-------------------------------");
                    System.out.println("Classification accuracy: " 
                                        + Math.round(performance * 10000.0) 
                                        / 100.0 + "%");
                    System.out.println();
                }
                else
                {
                    System.out.println(" ------------------------- ");
                    System.out.println("| CROSS VALIDATION FOLD " + (k+1) 
                                                                  + " |");
                    System.out.println(" ------------------------- ");
                    
                    // Build and train the RBF Network model
                    ArrayList<String> ypred = new ArrayList<String>();
                    
                    RBFNetwork rbf = new RBFNetwork(15, 0.95, 1000, 20);
                    rbf.fit(Xtrain, ytrain);
                    ypred = rbf.predict(Xtest);
                    
                    // Print the parameters and the prediction results
                    System.out.println();
                    System.out.println("Weights of each node");
                    System.out.println("--------------------");
                    
                    int j = 1;
                    for (RBFNetwork.Node node : rbf.nodes)
                    {
                        System.out.print("Hidden node " + j + ": [ ");
                        for (double weight : node.weights)
                        {
                            System.out.print(weight + " ");
                        }
                        System.out.println("]");
                        j++;
                    }
                    System.out.println();
                    
                    System.out.println("\tACTUAL" + "\t\tPREDICTED");
                    System.out.println("\t------" + "\t\t---------");
                    double performance = 0;
                    for (int i = 0; i < ypred.size(); i++)
                    {
                        System.out.println("\t" + ytest.get(i) + "\t| " 
                                                + ypred.get(i));
                        if (ytest.get(i).equals(ypred.get(i)))
                        {
                            performance++;
                        }
                    }
                    performance = performance / ytest.size();
                    System.out.println("-------------------------------");
                    System.out.println("Classification accuracy: " 
                                        + Math.round(performance * 10000.0) 
                                        / 100.0 + "%");
                    System.out.println();
                }
            }
        }
        
        scan.close();
    }
}
    
