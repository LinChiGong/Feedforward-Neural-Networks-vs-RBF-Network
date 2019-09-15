/**
 * This class reads data from file and preprocesses the data into the format
 * required by the "FeedForwardNetwork" class and the "RBFNetwork" class. The
 * method split() splits the data into 5 folds that can be used for cross
 * validation
 * 
 * @author Winston Lin
 */

import java.io.*;
import java.util.*;

public class ETL 
{    
    String fileName; // Name of the file, extracted from file path
    HashMap<String, Integer> classCount = 
            new HashMap<String, Integer>(); /* Count each class in order to do
                                               stratified split */
    boolean normalize = true;
    
    public ETL()
    {
    }
    
    public ETL(boolean normalize)
    {
        this.normalize = normalize;
    }
    
    /**
     * This method reads the 5 datasets used in this project, performs
     * appropriate processing, and stores each dataset in a 2D String array
     * 
     * @param filePath is the path of the data file
     * @return the 2D String array that stores the dataset
     * @throws FileNotFoundException
     */
    public ArrayList<String[]> readCSV(String filePath) 
            throws FileNotFoundException
    {
        fileName = filePath.substring(filePath.lastIndexOf('/') + 1,
                filePath.lastIndexOf('.'));
        
        ArrayList<String[]> records = new ArrayList<String[]>();
        String[] record;
        File file = new File(filePath);
        Scanner sc = new Scanner(file);
        while (sc.hasNextLine())
        {
            record = sc.nextLine().split(",", -1);
            
            if (fileName.equals("house-votes-84"))
            {
                for (int i = 0; i < record.length - 1; i++)
                {// Move the target column to the last position
                    String temp = record[i];
                    record[i] = record[i + 1];
                    record[i + 1] = temp;
                }
                records.add(record);
            }
            else if (fileName.equals("breast-cancer-wisconsin") || 
                    fileName.equals("glass"))
            {
                String[] newRecord = new String[record.length - 1];
                for (int i = 0; i < newRecord.length; i++)
                {// Ignore the first columns which contain id numbers
                    newRecord[i] = record[i + 1];
                }
                records.add(newRecord);
            }
            else
            {
                records.add(record);
            }
        }
        
        // Process each dataset
        if (fileName.equals("soybean-small"))
        {
            records = processSoybean(records); 
        }
        else if (fileName.equals("house-votes-84"))
        {     
            records = processVotes(records);
        }
        else if (fileName.equals("breast-cancer-wisconsin"))
        {     
            records = processCancer(records);
        }
        else 
        {
            if (normalize)
            {
                records = normalize(records);
            }
        }
        sc.close();
        
        return records;
    }
    
    /**
     * This method splits the data array into 5 partitions which can be used in
     * cross validation. The split can be random or stratified
     * 
     * @param records is the 2D String array that stores the dataset
     * @param stratified can be turned on to perform stratified split
     * @return the 5 partitions in an array
     */
    public ArrayList<ArrayList<String[]>> split(ArrayList<String[]> records,
            boolean stratified)
    {
        ArrayList<ArrayList<String[]>> partitions = 
                new ArrayList<ArrayList<String[]>>();
        
        // Count the number of instances in each class
        if (stratified)
        {
            for (int i = 0; i < records.size(); i++)
            {
                if (classCount.containsKey(
                        records.get(i)[records.get(0).length - 1]))
                {
                    int count = classCount.get(
                            records.get(i)[records.get(0).length - 1]);
                    count++;
                    classCount.put(
                            records.get(i)[records.get(0).length - 1], count);
                }
                else
                {
                    classCount.put(
                            records.get(i)[records.get(0).length - 1], 1);
                }
            }
        }
        
        // Split the data into 5 folds in preparation for cross validation
        int foldSize = records.size() / 5;
        for (int i = 0; i < 5; i++)
        {
            ArrayList<String[]> fold = new ArrayList<String[]>();
            
            if (stratified)
            {// Perform stratified split
                Iterator<Map.Entry<String, Integer>> it = 
                        classCount.entrySet().iterator();
                while(it.hasNext())
                {// For each class, select enough points to current fold
                    Map.Entry<String, Integer> pair = it.next();
                    int classPoints = pair.getValue();
                    if (i == 4)
                    {// Add all remaining data points to the last fold
                        while (records.size() != 0)
                        {
                            fold.add(records.get(0));
                            records.remove(0);
                        }
                        break;
                    }
                    // Make sure we have enough data points in each fold
                    int minPoints = classPoints / 5;
                    if (minPoints < 1)
                    {
                        minPoints = 1;
                    }
                    int j = 0;
                    while(j < records.size() && minPoints > 0)
                    {
                        if (pair.getKey().equals(
                                records.get(j)[records.get(0).length - 1]))
                        {
                            fold.add(records.get(j));
                            records.remove(j);
                            minPoints--;
                        }
                        j++;
                    }
                }
            }
            else
            {// Perform random split
                if (i == 4)
                {// Add all remaining data points to the last fold
                    while (records.size() != 0)
                    {
                        fold.add(records.get(0));
                        records.remove(0);
                    }
                }
                else
                {// Randomly select data points to add to the current fold
                    while(fold.size() < foldSize)
                    {
                        int randomIndex = (int) (Math.random()*records.size());
                        fold.add(records.get(randomIndex));
                        records.remove(randomIndex);
                    }
                }
            }
            partitions.add(fold);
        }
        
        return partitions;
    }
    
    /**
     * This method normalizes each feature in the dataset to be in range [0, 1]
     */
    public ArrayList<String[]> normalize(ArrayList<String[]> records)
    {        
        for (int i = 0; i < records.get(0).length - 1; i++)
        {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (String[] record : records)
            {
                if (Double.parseDouble(record[i]) < min)
                {
                    min = Double.parseDouble(record[i]);
                }
                if (Double.parseDouble(record[i]) > max)
                {
                    max = Double.parseDouble(record[i]);
                }
            }
            
            for (int j = 0; j < records.size(); j++)
            {// Normalize all features to range from 0 to 1
                double original = Double.parseDouble(records.get(j)[i]);
                if (max - min < 0.0001)
                {// When all values in a column are the same, set them to 0.5
                    records.get(j)[i] = Double.toString(0.5);
                }
                else
                {
                    double normalized = (original - min) / (max - min);
                    records.get(j)[i] = Double.toString(normalized);
                }
            }
        }
        
        return records;
    }
    
    /**
     * This method removes noninformative features in the "Soybean" dataset and
     * normalizes the dataset
     * 
     * @param records is the 2D String array that stores the "Soybean" dataset
     * @return the processed "Soybean" dataset in another 2D String array
     */
    public ArrayList<String[]> processSoybean(ArrayList<String[]> records)
    {
        ArrayList<String[]> newRecords = new ArrayList<String[]>();
        String[] newRecord;
        int length = 0; // Length of the informative features
        ArrayList<ArrayList<String>> categories = 
                new ArrayList<ArrayList<String>>();
        
        // Get categories for each feature
        for (int i = 0; i < records.get(0).length - 1; i++)
        {
            ArrayList<String> feature = new ArrayList<String>();
            for (int j = 0; j < records.size(); j++)
            {
                feature.add(records.get(j)[i]);
            } 
            ArrayList<String> uniqueValues = getUnique(feature);
            if (uniqueValues.size() > 1)
            {
                length += 1;
            }
            categories.add(uniqueValues);
        }
        
        // Remove noninformative features which have 0 variance
        for (int i = 0; i < records.size(); i++)
        {
            newRecord = new String[length + 1];
            int position = 0;
            for (int j = 0; j < records.get(0).length - 1; j++)
            {
                if (categories.get(j).size() == 1)
                {
                    continue;
                }
                
                newRecord[position] = records.get(i)[j];

                if (categories.get(j).size() != 1)
                {
                    position += 1;
                }
            }
            newRecord[newRecord.length - 1] = 
                    records.get(i)[records.get(0).length - 1];
            newRecords.add(newRecord);
        }
        
        // Normalize the reduced dataset
        if (normalize)
        {
            newRecords = normalize(newRecords);
        }
        
        return newRecords;
    }
    
    /**
     * This method processes the "Vote" dataset
     * 
     * @param records is the 2D String array that stores the "Vote" dataset
     * @return the processed "Vote" dataset in another 2D String array
     */
    public ArrayList<String[]> processVotes(ArrayList<String[]> records)
    {
        ArrayList<String[]> newRecords = new ArrayList<String[]>();
        String[] newRecord;
        
        for (int i = 0; i < records.size(); i++)
        {
            newRecord = new String[records.get(0).length];
            for (int j = 0; j < records.get(0).length - 1; j++)
            {
                // Transform the features
                if (records.get(i)[j].equals("y"))
                {
                    newRecord[j] = "1";
                }
                else if (records.get(i)[j].equals("n"))
                {
                    newRecord[j] = "0";
                }
                else
                {/* Fill in the missing values with the majority vote from all
                    class members */
                    int classYes = 0;
                    int classNo = 0;
                    for (int k = 0; k < records.size(); k++)
                    {
                        if (records.get(k)[records.get(0).length - 1].equals(
                                records.get(i)[records.get(0).length - 1]))
                        {
                            if (records.get(k)[j].equals("y"))
                            {
                                classYes++;
                            }
                            else if (records.get(k)[j].equals("n"))
                            {
                                classNo++;
                            }
                        }
                    }
                    if (classYes > classNo)
                    {
                        newRecord[j] = "1";
                    }
                    else
                    {
                        newRecord[j] = "0";
                    }
                }
            }
            newRecord[newRecord.length - 1] = 
                    records.get(i)[records.get(0).length - 1];
            newRecords.add(newRecord);
        }
        
        return newRecords;
    }
    
    /**
     * This method processes and normalizes the "Breast Cancer" dataset
     * 
     * @param records is the 2D String array that stores the dataset
     * @return the processed "Breast Cancer" dataset in another 2D String array
     */
    public ArrayList<String[]> processCancer(ArrayList<String[]> records)
    {
        ArrayList<String[]> newRecords = new ArrayList<String[]>();
        
        // Fill in the missing values with the median of all class members
        for (int i = 0; i < records.size(); i++)
        {
            for (int j = 0; j < records.get(0).length - 1; j++)
            {
                if (records.get(i)[j].equals("?"))
                {
                    int classMedian = getMedian(records, j, 
                            records.get(i)[records.get(0).length - 1]);
                    records.get(i)[j] = Integer.toString(classMedian);
                }
            }
        }
        
        // Normalize the dataset
        if (normalize)
        {
            newRecords = normalize(records);
        }
        
        return newRecords;
    }

    /**
     * This method finds all unique values in an array
     * 
     * @param x is the array of interest
     * @return a new array without redundant values
     */
    public ArrayList<String> getUnique(ArrayList<String> x)
    {
        ArrayList<String> uniqueValues = new ArrayList<String>();

        for(int i = 0; i < x.size(); i++)
        {
            if(!uniqueValues.contains(x.get(i)))
            {
                uniqueValues.add(x.get(i));
            }
        }

       return uniqueValues;
    }
    
    /**
     * This method finds the median value among members of a certain class
     * 
     * @param records is the 2D String array that stores the dataset
     * @param col is the feature index
     * @param label is the class of interest
     * @return the median value
     */
    public int getMedian(ArrayList<String[]> records, int col, String label)
    {
        int median;
        ArrayList<Integer> classMember = new ArrayList<Integer>();
        for (int i = 0; i < records.size(); i++)
        {
            if (records.get(i)[records.get(0).length - 1].equals(label) && 
                    !records.get(i)[col].equals("?"))
            {
                classMember.add(Integer.parseInt(records.get(i)[col]));
            }
        }
        Collections.sort(classMember);
        median = classMember.get(classMember.size() / 2);
        
        return median;
    }
}
