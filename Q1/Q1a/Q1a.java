import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount2 {

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        private boolean caseSensitive = false;
        private Set<String> patternsToSkip = new HashSet<>();

        @Override
        public void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            caseSensitive = conf.getBoolean("wordcount.case.sensitive", false);

            URI[] cacheFiles = context.getCacheFiles();

            if (cacheFiles != null) {
                for (URI uri : cacheFiles) {
                    BufferedReader reader = new BufferedReader(new FileReader(new java.io.File(uri.getPath())));
                    String pattern;
                    while ((pattern = reader.readLine()) != null) {
                        patternsToSkip.add(pattern.trim().toLowerCase());
                    }
                    reader.close();
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = caseSensitive ? value.toString() : value.toString().toLowerCase();

            String[] tokens = line.split("[^a-zA-Z]+");

            for (String token : tokens) {
                if (token.length() >= 2 && !patternsToSkip.contains(token)) {
                    word.set(token);
                    context.write(word, one); 
                }
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private Map<String, Integer> countMap = new HashMap<>();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) {

            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            countMap.put(key.toString(), sum);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {

            List<Map.Entry<String, Integer>> list = new ArrayList<>(countMap.entrySet());

            list.sort((a, b) -> b.getValue() - a.getValue());

            int count = 0;
            for (Map.Entry<String, Integer> entry : list) {
                if (count == 50) break;
                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
                count++;
            }
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Top 50 Words using Pairs Approach");

        job.setJarByClass(WordCount2.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setNumReduceTasks(1);

        for (int i = 0; i < args.length; ++i) {
            if ("-skippatterns".equals(args[i])) {
                job.getConfiguration().setBoolean("wordcount.skip.patterns", true);
                job.addCacheFile(new Path(args[++i]).toUri());
            } else if ("-casesensitive".equals(args[i])) {
                job.getConfiguration().setBoolean("wordcount.case.sensitive", true);
            }
        }

        FileInputFormat.addInputPath(job, new Path(args[args.length - 2]));
        FileOutputFormat.setOutputPath(job, new Path(args[args.length - 1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}