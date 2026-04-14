import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Pairs Co-occurrence — MAP-CLASS LEVEL local aggregation + Combiner.
 *
 * HashMap `counts` is a CLASS FIELD — persists across ALL map() calls.
 * Flushes ONCE in cleanup() — fewest intermediate pairs sent to combiner.
 * Combiner then further reduces before sending to reducer.
 *
 * Usage:
 *   hadoop jar PairsClassLevel.jar PairsClassLevel <input> <o> <window_d> <top50words>
 */
public class PairsClassLevel {

    public static class PairMapper extends Mapper<Object, Text, Text, IntWritable> {

        // ── CLASS-LEVEL buffer: lives for the entire mapper task ──
        private Map<String, Integer> counts = new HashMap<>();
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

                int left  = Math.max(0, i - distance);
                int right = Math.min(tokens.length - 1, i + distance);

                for (int j = left; j <= right; j++) {
                    if (j == i) continue;
                    String u = tokens[j];
                    if (u.isEmpty() || !topWords.contains(u)) continue;

                    // Accumulate locally — NO context.write here
                    String pair = w + "," + u;
                    counts.merge(pair, 1, Integer::sum);
                }
            }
        }

        // Flush ONCE after all map() calls for this task
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                context.write(new Text(entry.getKey()),
                              new IntWritable(entry.getValue()));
            }
        }
    }

    // Combiner = same as Reducer (summing is associative, safe to use as combiner)
    public static class PairCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) sum += val.get();
            context.write(key, new IntWritable(sum));
        }
    }

    public static class PairReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) sum += val.get();
            context.write(key, new IntWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.setInt("window", Integer.parseInt(args[2]));

        Job job = Job.getInstance(conf, "Pairs Class-Level + Combiner d=" + args[2]);
        job.setJarByClass(PairsClassLevel.class);
        job.setMapperClass(PairMapper.class);
        job.setCombinerClass(PairCombiner.class);   // ← COMBINER ADDED
        job.setReducerClass(PairReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.addCacheFile(new Path(args[3]).toUri());
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}