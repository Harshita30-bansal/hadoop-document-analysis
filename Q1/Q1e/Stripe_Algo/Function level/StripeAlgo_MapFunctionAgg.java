import java.io.*;
import java.util.*;

import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.w3c.dom.Text;

public class StripeAlgo_MapFunctionAgg {

    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> topWords = new HashSet<>();
        private int d;
        private Text wordKey = new Text();

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            d = conf.getInt("distance", 1);

            Path[] localPaths = context.getLocalCacheFiles();
            BufferedReader reader = new BufferedReader(new FileReader(localPaths[0].toString()));

            String line;
            while ((line = reader.readLine()) != null) {
                String word = line.trim().split("\\s+")[0];
                topWords.add(word);
            }
            reader.close();
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

                    Text neighbor = new Text(u);

                    if (stripe.containsKey(neighbor)) {
                        IntWritable count = (IntWritable) stripe.get(neighbor);
                        count.set(count.get() + 1);
                    } else {
                        stripe.put(neighbor, new IntWritable(1));
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

        Job job = Job.getInstance(conf, "Stripes Map-Function Aggregation");

        job.setJarByClass(StripeAlgo_MapFunctionAgg.class);
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

        long start = System.currentTimeMillis();
        boolean success = job.waitForCompletion(true);
        long end = System.currentTimeMillis();

        long duration = end - start;
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