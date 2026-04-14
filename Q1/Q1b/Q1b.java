import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Pairs {

    public static class PairMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Set<String> topWords = new HashSet<>();
        private int distance;

        @Override
        protected void setup(Context context) throws IOException {

            Configuration conf = context.getConfiguration();
            distance = conf.getInt("window", 1);

            URI[] cacheFiles = context.getCacheFiles();

            if (cacheFiles != null && cacheFiles.length > 0) {
                String fileName = new File(cacheFiles[0].getPath()).getName();
                BufferedReader reader = new BufferedReader(new FileReader(fileName));

                String line;
                while ((line = reader.readLine()) != null) {
                    topWords.add(line.trim().toLowerCase());
                }
                reader.close();
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

            for (int i = 0; i < tokens.length; i++) {

                String w = tokens[i];

                if (w.isEmpty() || !topWords.contains(w)) continue;

                int left = Math.max(0, i - distance);
                int right = Math.min(tokens.length - 1, i + distance);

                for (int j = left; j <= right; j++) {

                    if (j == i) continue;

                    String u = tokens[j];

                    if (u.isEmpty() || !topWords.contains(u)) continue;

                    String pair = w + "," + u;
                    context.write(new Text(pair), one);
                }
            }
        }
    }

    public static class PairReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            context.write(key, new IntWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();

        conf.setInt("window", Integer.parseInt(args[2]));

        Job job = Job.getInstance(conf, "Pairs Co-occurrence");

        job.setJarByClass(Pairs.class);
        job.setMapperClass(PairMapper.class);
        job.setReducerClass(PairReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.addCacheFile(new Path(args[3]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}