import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class StripeAlgo_MapClassAgg {

    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> topWords = new HashSet<>();
        private int d;

        private Map<String, MapWritable> globalStripes = new HashMap<>();

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
                throws IOException {

            String[] tokens = value.toString().toLowerCase().split("[^\\w']+");

            for (int i = 0; i < tokens.length; i++) {
                String w = tokens[i];
                if (w.isEmpty() || !topWords.contains(w)) continue;

                MapWritable stripe = globalStripes.getOrDefault(w, new MapWritable());

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

                globalStripes.put(w, stripe);
            }
        }

        @Override
        protected void cleanup(Context context)
                throws IOException, InterruptedException {

            for (Map.Entry<String, MapWritable> entry : globalStripes.entrySet()) {
                context.write(new Text(entry.getKey()), entry.getValue());
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

        Job job = Job.getInstance(conf, "Stripes Map-Class Aggregation");

        job.setJarByClass(StripeAlgo_MapClassAgg.class);
        job.setMapperClass(StripeMapper.class);

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