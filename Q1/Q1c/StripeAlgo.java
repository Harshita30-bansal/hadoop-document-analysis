import java.io.*;
import java.net.URI;
import java.util.*;

import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.Path;

public class StripeAlgo {

    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> topWords = new HashSet<>();
        private int d;
        private Text wordKey = new Text();
        private Text neighborKey = new Text();

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            d = conf.getInt("distance", 1);

            Path[] localPaths = context.getLocalCacheFiles();
            if (localPaths != null && localPaths.length > 0) {
                BufferedReader reader = new BufferedReader(
                    new FileReader(localPaths[0].toString()));

                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty()) {
                        String word = line.split("\\s+")[0];
                        topWords.add(word);
                    }
                }
                reader.close();
            }

            if (topWords.isEmpty()) {
                throw new IOException("Top words file loaded empty!");
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("[^\\w']+");

            for (int i = 0; i < tokens.length; i++) {
                String w = tokens[i];

                if (w.isEmpty() || !topWords.contains(w)) continue;

                MapWritable stripe = new MapWritable();

                int left = Math.max(0, i - d);
                int right = Math.min(tokens.length - 1, i + d);

                for (int j = left; j <= right; j++) {
                    if (j == i) continue;

                    String u = tokens[j];
                    if (u.isEmpty() || !topWords.contains(u)) continue;

                    neighborKey.set(u);

                    if (stripe.containsKey(neighborKey)) {
                        IntWritable count = (IntWritable) stripe.get(neighborKey);
                        count.set(count.get() + 1);
                    } else {
                        stripe.put(new Text(u), new IntWritable(1));
                    }
                }

                if (!stripe.isEmpty()) {
                    wordKey.set(w);
                    context.write(wordKey, stripe);
                }
            }
        }
    }

    public static class StripeReducer extends Reducer<Text, MapWritable, Text, MapWritable> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            MapWritable result = new MapWritable();

            for (MapWritable stripe : values) {
                for (Writable k : stripe.keySet()) {
                    IntWritable val = (IntWritable) stripe.get(k);

                    if (result.containsKey(k)) {
                        IntWritable existing = (IntWritable) result.get(k);
                        existing.set(existing.get() + val.get());
                    } else {
                        result.put(new Text(k.toString()), new IntWritable(val.get()));
                    }
                }
            }

            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        conf.setInt("distance", Integer.parseInt(args[2]));
        conf.set("mapreduce.input.fileinputformat.split.maxsize", "268435456");

        Job job = Job.getInstance(conf, "Stripes CoOccurrence");

        job.setJarByClass(StripeAlgo.class);
        job.setMapperClass(StripeMapper.class);
        job.setCombinerClass(StripeReducer.class);
        job.setReducerClass(StripeReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(MapWritable.class);

        job.addCacheFile(new Path(args[3]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        long startTime = System.currentTimeMillis();
        boolean success = job.waitForCompletion(true);
        long endTime = System.currentTimeMillis();

        long duration = endTime - startTime;
        long seconds = duration / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;

        seconds = seconds % 60;
        minutes = minutes % 60;

        System.out.println("Runtime (ms): " + duration);
        System.out.println("Runtime: " + hours + "h " + minutes + "m " + seconds + "s");

        System.exit(success ? 0 : 1);
    }
}