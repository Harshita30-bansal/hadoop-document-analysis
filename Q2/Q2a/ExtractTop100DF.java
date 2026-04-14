import java.io.*;
import java.util.*;

/**
 * ExtractTop100DF.java
 *
 * Run this locally (plain java, NOT hadoop jar) after DocumentFrequency job finishes.
 *
 * What it does:
 *   1. Reads the Hadoop output file (part-r-00000)
 *      Each line looks like:   system    4821
 *   2. Sorts all terms by DF descending (highest first)
 *   3. Prints top 100 in a ranked table on the terminal
 *   4. Saves top 100 as top100df.tsv  (TERM<TAB>DF)
 *      This file is used as distributed cache input in Problem 2b
 *
 * Compile: javac ExtractTop100DF.java
 * Run:     java ExtractTop100DF output_2a/part-r-00000
 */
public class ExtractTop100DF {

    public static void main(String[] args) throws IOException {

        if (args.length < 1) {
            System.err.println("Usage: java ExtractTop100DF <part-r-00000 path>");
            System.exit(1);
        }

        // Read all term-DF pairs from the MapReduce output file
        List<String[]> termDfList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(args[0]));
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] parts = line.split("\t");
            if (parts.length == 2) {
                try {
                    Integer.parseInt(parts[1].trim()); // make sure it is a number
                    termDfList.add(new String[]{ parts[0].trim(), parts[1].trim() });
                } catch (NumberFormatException e) {
                    // skip any malformed lines
                }
            }
        }
        reader.close();

        System.out.println("Total unique stemmed terms found: " + termDfList.size());

        // Sort by DF descending — highest document frequency first
        Collections.sort(termDfList, new Comparator<String[]>() {
            @Override
            public int compare(String[] a, String[] b) {
                return Integer.compare(
                    Integer.parseInt(b[1]),
                    Integer.parseInt(a[1])
                );
            }
        });

        // Print top 100 as a neat table
        int limit = Math.min(100, termDfList.size());
        System.out.println("\n========= TOP 100 TERMS BY DOCUMENT FREQUENCY =========");
        System.out.printf("%-6s  %-25s  %s%n", "Rank", "Term (stemmed)", "Doc Frequency");
        System.out.println("--------------------------------------------------------");
        for (int i = 0; i < limit; i++) {
            System.out.printf("%-6d  %-25s  %s%n",
                (i + 1), termDfList.get(i)[0], termDfList.get(i)[1]);
        }

        // Save top 100 to TSV file — needed for Problem 2b distributed cache
        String outFile = "top100df.tsv";
        BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
        for (int i = 0; i < limit; i++) {
            writer.write(termDfList.get(i)[0] + "\t" + termDfList.get(i)[1]);
            writer.newLine();
        }
        writer.close();

        System.out.println("\nSaved to: " + outFile);
        System.out.println("Next: hdfs dfs -put top100df.tsv /wiki/resources/top100df.tsv");
    }
}
