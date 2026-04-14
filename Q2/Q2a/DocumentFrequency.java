import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.util.GenericOptionsParser;

import opennlp.tools.stemmer.PorterStemmer;

public class DocumentFrequency {

    // =========================================================================
    // MAPPER
    // =========================================================================
    // Same logic as before — tokenize, stem, filter stopwords, emit (word, 1)
    // The only difference is CombineTextInputFormat is now used in the driver
    // so instead of 10,000 splits we get ~80 splits of 128MB each
    // This completely fixes the OutOfMemoryError
    // =========================================================================
    public static class DFMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Set<String> stopWords = new HashSet<>();
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private PorterStemmer stemmer = new PorterStemmer();

        @Override
        protected void setup(Context context) throws IOException {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null && cacheFiles.length > 0) {
                BufferedReader br = new BufferedReader(new FileReader("stopwords.txt"));
                String line;
                while ((line = br.readLine()) != null) {
                    stopWords.add(line.trim().toLowerCase());
                }
                br.close();
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("\\W+");

            // HashSet deduplicates within one line so same word on same line
            // only counts once toward document frequency
            Set<String> uniqueTerms = new HashSet<>();

            for (String token : tokens) {
                token = token.trim();
                if (token.isEmpty()) continue;
                if (!token.matches("[a-zA-Z]+")) continue;
                if (stopWords.contains(token)) continue;
                token = stemmer.stem(token).toString();
                uniqueTerms.add(token);
            }

            for (String term : uniqueTerms) {
                word.set(term);
                context.write(word, one);
            }
        }
    }

    // =========================================================================
    // REDUCER
    // =========================================================================
    // Sums all the 1s for each word = Document Frequency
    // =========================================================================
    public static class DFReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    // =========================================================================
    // DRIVER
    // =========================================================================
    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        // Need 3 args: input, output, stopwords path
        if (otherArgs.length < 3) {
            System.err.println(
                "Usage: DocumentFrequency <input_path> <output_path> <stopwords_hdfs_path>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "Document Frequency 2a");

        job.setJarByClass(DocumentFrequency.class);
        job.setMapperClass(DFMapper.class);
        job.setReducerClass(DFReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // THE KEY FIX:
        // Default FileInputFormat creates 1 split per file = 10,000 splits
        // Hadoop then creates 10,000 MapTaskRunnable objects in RAM at once
        // = OutOfMemoryError before any processing even starts
        //
        // CombineTextInputFormat merges multiple small files into one split
        // 128MB per split means ~80 splits total from 10,000 files
        // Hadoop only creates ~80 MapTaskRunnable objects = no OOM
        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 134217728); // 128MB

        // Add stopwords.txt to distributed cache
        // otherArgs[2] is the HDFS path to stopwords.txt
        job.addCacheFile(new Path(otherArgs[2]).toUri());

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}