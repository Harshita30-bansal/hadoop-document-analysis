import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;

import opennlp.tools.stemmer.PorterStemmer;

/**
 * Problem 2b - TF-IDF Scoring using Stripes approach
 *
 * FORMULA: SCORE = TF x log(10000 / DF + 1)
 *
 * OUTPUT FORMAT: ID<tab>TERM<tab>SCORE
 *   where ID = document filename (e.g. "563203.txt")
 *
 * STRIPES APPROACH:
 *   Mapper processes ONE document at a time (using CustomFileInputFormat).
 *   For each document, counts TF of each top-100 term locally in a HashMap.
 *   In cleanup(), emits ONE stripe per document:
 *     key   = filename (document ID)
 *     value = "term1:count1,term2:count2,..."
 *   Reducer loads DF values, computes SCORE, outputs ID<TAB>TERM<TAB>SCORE.
 *
 * WHY CustomFileInputFormat:
 *   Unlike 2a which just needed word counts (no need to know which file),
 *   2b needs per-document TF scores so we MUST know the filename.
 *   CustomFileInputFormat (given file, unchanged) gives us FileSplit
 *   which contains the filename = document ID.
 */
public class TFIDFScore {

    // =========================================================================
    // MAPPER — Stripes approach, one document at a time
    // =========================================================================
    public static class TFIDFMapper extends Mapper<Object, Text, Text, Text> {

        private PorterStemmer stemmer   = new PorterStemmer();
        private Set<String>   stopWords = new HashSet<>();

        // Top 100 terms and their DF values — loaded from top100df.tsv
        private Map<String, Integer> top100DF = new HashMap<>();

        // Local TF counter — counts occurrences of top-100 terms in THIS document
        // Resets automatically per mapper task (one mapper = one document)
        private Map<String, Integer> localTF = new HashMap<>();

        // The filename of the document being processed = document ID
        private String docId = "";

        private Text outKey   = new Text();
        private Text outValue = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {

            // Get document filename from FileSplit
            // CustomFileInputFormat gives us a FileSplit per file
            // so getName() gives us e.g. "563203.txt"
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            docId = fileSplit.getPath().getName();

            // Load both cached files
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI uri : cacheFiles) {
                    String fileName = new Path(uri.getPath()).getName();
                    if (fileName.equals("stopwords.txt")) {
                        loadStopWords(fileName);
                    } else if (fileName.equals("top100df.tsv")) {
                        loadTop100DF(fileName);
                    }
                }
            }
        }

        private void loadStopWords(String fileName) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(fileName));
                String line;
                while ((line = reader.readLine()) != null) {
                    String sw = line.trim().toLowerCase();
                    if (!sw.isEmpty()) stopWords.add(sw);
                }
                reader.close();
            } catch (IOException e) {
                System.err.println("Stopwords error: " + StringUtils.stringifyException(e));
            }
        }

        private void loadTop100DF(String fileName) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(fileName));
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    String[] parts = line.split("\t");
                    if (parts.length == 2) {
                        try {
                            top100DF.put(parts[0].trim(), Integer.parseInt(parts[1].trim()));
                        } catch (NumberFormatException e) {}
                    }
                }
                reader.close();
            } catch (IOException e) {
                System.err.println("top100df error: " + StringUtils.stringifyException(e));
            }
        }

        // Called for EVERY LINE of the document
        // Counts occurrences of top-100 terms in this document
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-z0-9]+");

            for (String token : tokens) {
                if (token.isEmpty()) continue;
                if (token.matches("[0-9]+")) continue;
                if (token.length() < 2) continue;
                if (stopWords.contains(token)) continue;

                String stemmed = stemmer.stem(token);
                if (stemmed == null || stemmed.length() < 2) continue;

                // Only count if it is one of the top-100 terms
                if (top100DF.containsKey(stemmed)) {
                    localTF.put(stemmed, localTF.getOrDefault(stemmed, 0) + 1);
                }
            }
        }

        // Called ONCE after ALL lines of this document are processed
        // This is the STRIPES emit — one output per document
        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            if (localTF.isEmpty()) return;

            // Build stripe: "term1:count1,term2:count2,..."
            StringBuilder stripe = new StringBuilder();
            for (Map.Entry<String, Integer> entry : localTF.entrySet()) {
                if (stripe.length() > 0) stripe.append(",");
                stripe.append(entry.getKey()).append(":").append(entry.getValue());
            }

            // Emit: (documentFilename, stripe)
            // One emission per document = stripes approach
            outKey.set(docId);
            outValue.set(stripe.toString());
            context.write(outKey, outValue);
        }
    }

    // =========================================================================
    // REDUCER
    // =========================================================================
    // Receives: (docID, [stripe1, stripe2, ...])
    // Merges term counts from all stripes
    // Computes: SCORE = TF x log(10000 / DF + 1)
    // Outputs:  docID<TAB>term<TAB>score
    // =========================================================================
    public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {

        private Map<String, Integer> top100DF = new HashMap<>();
        private Text outValue = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI uri : cacheFiles) {
                    String fileName = new Path(uri.getPath()).getName();
                    if (fileName.equals("top100df.tsv")) {
                        loadTop100DF(fileName);
                    }
                }
            }
        }

        private void loadTop100DF(String fileName) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(fileName));
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    String[] parts = line.split("\t");
                    if (parts.length == 2) {
                        try {
                            top100DF.put(parts[0].trim(), Integer.parseInt(parts[1].trim()));
                        } catch (NumberFormatException e) {}
                    }
                }
                reader.close();
            } catch (IOException e) {
                System.err.println("top100df reducer error: " + StringUtils.stringifyException(e));
            }
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            String docId = key.toString();

            // Merge all stripes for this document
            Map<String, Integer> mergedTF = new HashMap<>();
            for (Text val : values) {
                String[] pairs = val.toString().split(",");
                for (String pair : pairs) {
                    String[] termCount = pair.split(":");
                    if (termCount.length == 2) {
                        try {
                            String term  = termCount[0].trim();
                            int    count = Integer.parseInt(termCount[1].trim());
                            mergedTF.put(term, mergedTF.getOrDefault(term, 0) + count);
                        } catch (NumberFormatException e) {}
                    }
                }
            }

            // Compute SCORE for each term and emit
            for (Map.Entry<String, Integer> entry : mergedTF.entrySet()) {
                String term = entry.getKey();
                int    tf   = entry.getValue();

                Integer df = top100DF.get(term);
                if (df == null || df == 0) continue;

                // SCORE = TF x log(10000 / DF + 1)
                double score = tf * Math.log((10000.0 / df) + 1);

                // Output: docID<TAB>term<TAB>score
                outValue.set(term + "\t" + String.format("%.6f", score));
                context.write(key, outValue);
            }
        }
    }

    // =========================================================================
    // DRIVER
    // =========================================================================
    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length < 4) {
            System.err.println(
                "Usage: TFIDFScore <input> <o> <stopwords_hdfs_path> <top100df_hdfs_path>");
            System.exit(1);
        }

        Job job = Job.getInstance(conf, "tfidf_score_2b");

        job.setJarByClass(TFIDFScore.class);
        job.setMapperClass(TFIDFMapper.class);
        job.setReducerClass(TFIDFReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // CustomFileInputFormat — gives mapper the filename (docID) via FileSplit
        // This is the key difference from 2a
        // Each file = one mapper task = one document ID
        job.setInputFormatClass(CustomFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Add both cache files — both mapper and reducer need top100df.tsv
        job.addCacheFile(new Path(otherArgs[2]).toUri()); // stopwords.txt
        job.addCacheFile(new Path(otherArgs[3]).toUri()); // top100df.tsv

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
