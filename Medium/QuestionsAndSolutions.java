package Medium;

import java.util.Random;
import java.util.Set;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

public class QuestionsAndSolutions {
    // S1.
    static class Node<T> {
        T val;
        Node<T> left;
        Node<T> right;

        public Node(T val, Node<T> left, Node<T> right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static String serialize(Node<String> root) {
        if (root == null) {
            return "null";
        }

        String leftSerialized = serialize(root.left);
        String rightSerialized = serialize(root.right);

        return root.val + "," + leftSerialized + "," + rightSerialized;
    }

    public static Node<String> deserialize(String data) {
        Queue<String> nodes = new LinkedList<>(Arrays.asList(data.split(",")));
        return deserializeHelper(nodes);
    }

    private static Node<String> deserializeHelper(Queue<String> nodes) {
        String val = nodes.poll();

        if (val.equals("null")) {
            return null;
        }

        Node<String> node = new Node<>(val, null, null);
        node.left = deserializeHelper(nodes);
        node.right = deserializeHelper(nodes);

        return node;
    }

    // S2.
    static class Pair<A, B> {
        private A first;
        private B second;

        public Pair(A first, B second) {
            this.first = first;
            this.second = second;
        }

        public A car() {
            return first;
        }

        public B cdr() {
            return second;
        }
    }

    public static <A, B> Pair<A, B> cons(A a, B b) {
        return new Pair<>(a, b);
    }

    public static <A, B> A car(Pair<A, B> pair) {
        return pair.car();
    }

    public static <A, B> B cdr(Pair<A, B> pair) {
        return pair.cdr();
    }

    // S3.
    public static int numDecodings(String message) {
        if (message == null || message.isEmpty()) {
            return 0;
        }

        int n = message.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;

        // Handle single-digit cases
        dp[1] = message.charAt(0) == '0' ? 0 : 1;

        for (int i = 2; i <= n; i++) {
            int oneDigit = Integer.parseInt(message.substring(i - 1, i));
            int twoDigits = Integer.parseInt(message.substring(i - 2, i));

            if (oneDigit >= 1 && oneDigit <= 9) {
                dp[i] += dp[i - 1];
            }

            if (twoDigits >= 10 && twoDigits <= 26) {
                dp[i] += dp[i - 2];
            }
        }

        return dp[n];
    }

    // S4.
    public static void scheduleJob(Runnable task, long delayMillis) {
        Thread schedulerThread = new Thread(() -> {
            try {
                Thread.sleep(delayMillis);
                task.run();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        schedulerThread.start();
    }

    // S5.
    static class TrieNode {
        private final TrieNode[] children;
        private boolean isEndOfWord;

        public TrieNode() {
            this.children = new TrieNode[26];
            this.isEndOfWord = false;
        }

        public TrieNode getChild(char ch) {
            return children[ch - 'a'];
        }

        public void setChild(char ch, TrieNode node) {
            children[ch - 'a'] = node;
        }

        public boolean isEndOfWord() {
            return isEndOfWord;
        }

        public void setEndOfWord(boolean endOfWord) {
            isEndOfWord = endOfWord;
        }
    }

    static class AutocompleteSystem {
        private final TrieNode root;

        public AutocompleteSystem() {
            this.root = new TrieNode();
        }

        public void insert(String word) {
            TrieNode current = root;
            for (char ch : word.toCharArray()) {
                TrieNode child = current.getChild(ch);
                if (child == null) {
                    child = new TrieNode();
                    current.setChild(ch, child);
                }
                current = child;
            }
            current.setEndOfWord(true);
        }

        public List<String> search(String prefix) {
            List<String> results = new ArrayList<>();
            TrieNode current = root;
            for (char ch : prefix.toCharArray()) {
                TrieNode child = current.getChild(ch);
                if (child == null) {
                    return results;
                }
                current = child;
            }
            collectWords(current, prefix, results);
            return results;
        }

        private void collectWords(TrieNode node, String prefix, List<String> results) {
            if (node.isEndOfWord()) {
                results.add(prefix);
            }
            for (char ch = 'a'; ch <= 'z'; ch++) {
                TrieNode child = node.getChild(ch);
                if (child != null) {
                    collectWords(child, prefix + ch, results);
                }
            }
        }
    }

    // S6.
    public static double estimatePieDecimalPlaces() {
        int totalPoints = 1000000;
        int pointsInsideCircle = 0;

        Random random = new Random();
        for (int i = 0; i < totalPoints; i++) {
            double x = random.nextDouble();
            double y = random.nextDouble();

            double distance = x * x + y * y;
            if (distance <= 1) {
                pointsInsideCircle++;
            }
        }

        return 4.0 * pointsInsideCircle / totalPoints;
    }

    // S7.
    static class RandomElementPicker<T> {
        private T currentElement;
        private int count;

        public void pickElement(T element) {
            Random random = new Random();
            count++;
            // The probability of updating the currentElement decreases as more elements are
            // encountered, ensuring a uniform selection.
            if (random.nextInt(count) == 0) {
                currentElement = element;
            }
        }

        public T getRandomElement() {
            return currentElement;
        }
    }

    // S8.
    public static int minCost(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }

        int n = costs.length;
        int k = costs[0].length;

        int[][] dp = new int[n][k];

        // Initialize the first row of the dp matrix with the costs of the first house
        for (int i = 0; i < k; i++) {
            dp[0][i] = costs[0][i];
        }

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < k; j++) {
                dp[i][j] = costs[i][j] + getMinCost(dp, i - 1, j);
            }
        }

        // Find the minimum cost among the last row
        int minCost = dp[n - 1][0];
        for (int i = 1; i < k; i++) {
            minCost = Math.min(minCost, dp[n - 1][i]);
        }

        return minCost;
    }

    // Helper method to get the minimum cost of painting the previous house with a
    // different color
    private static int getMinCost(int[][] dp, int row, int color) {
        int minCost = Integer.MAX_VALUE;
        for (int i = 0; i < dp[row].length; i++) {
            if (i != color) {
                minCost = Math.min(minCost, dp[row][i]);
            }
        }
        return minCost;
    }

    // S9.
    public static List<String> reconstructSentence(Set<String> wordDict, String sentence) {
        Map<String, List<String>> memo = new HashMap<>();
        return reconstruct(wordDict, sentence, memo);
    }

    private static List<String> reconstruct(Set<String> wordDict, String sentence, Map<String, List<String>> memo) {
        if (memo.containsKey(sentence)) {
            return memo.get(sentence);
        }

        List<String> result = new ArrayList<>();
        if (sentence.isEmpty()) {
            result.add("");
            return result;
        }

        for (String word : wordDict) {
            if (sentence.startsWith(word)) {
                String suffix = sentence.substring(word.length());
                List<String> subSentences = reconstruct(wordDict, suffix, memo);
                for (String subSentence : subSentences) {
                    String reconstructed = word + (subSentence.isEmpty() ? "" : " " + subSentence);
                    result.add(reconstructed);
                }
            }
        }

        memo.put(sentence, result);
        return result;
    }

    // S10.
    static class LockingTreeNode {
        private boolean locked;
        private int lockedDescendantsCount;
        private LockingTreeNode parent;
        private LockingTreeNode left;
        private LockingTreeNode right;

        public boolean isLocked() {
            return locked;
        }

        public boolean lock() {
            if (lockedDescendantsCount > 0 || hasLockedAncestor()) {
                return false;
            }

            locked = true;

            updateLockedDescendantsCount(1);

            return true;
        }

        public boolean unlock() {
            if (lockedDescendantsCount > 0 || hasLockedAncestor()) {
                return false;
            }

            locked = false;

            updateLockedDescendantsCount(-1);

            return true;
        }

        private boolean hasLockedAncestor() {
            LockingTreeNode current = parent;
            while (current != null) {
                if (current.isLocked()) {
                    return true;
                }
                current = current.parent;
            }
            return false;
        }

        private void updateLockedDescendantsCount(int count) {
            LockingTreeNode current = parent;
            while (current != null) {
                current.lockedDescendantsCount += count;
                current = current.parent;
            }
        }
    }

    public static void main(String[] args) {
        /*
         * Q1.
         * Given the root to a binary tree, implement serialize(root), which serializes
         * the tree into a string, and deserialize(s), which deserializes the string
         * back into the tree.
         * For example, given the following Node class
         * "class Node:                                            "
         * "    def __init__(self, val, left=None, right=None):    "
         * "        self.val = val                                 "
         * "        self.left = left                               "
         * "        self.right = right                             "
         * The following test should pass:
         * node = Node('root', Node('left', Node('left.left')), Node('right'))
         * assert deserialize(serialize(node)).left.left.val == 'left.left'
         */
        System.out.println("========= Q1 ==========");
        Node<String> node = new Node<>("root", new Node<>("left", new Node<>("left.left", null, null), null),
                new Node<>("right", null, null));
        String serialized = serialize(node);
        System.out.println("Serialized tree: " + serialized);
        Node<String> deserialized = deserialize(serialized);
        System.out.println("Deserialized value: " + deserialized.left.left.val);

        /*
         * Q2.
         * cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first
         * and last element of that pair. For example, car(cons(3, 4)) returns 3, and
         * cdr(cons(3, 4)) returns 4.
         * Given this implementation of cons:
         * "def cons(a, b):        "
         * "    def pair(f):       "
         * "        return f(a, b) "
         * "    return pair        "
         * Implement car and cdr.
         */
        System.out.println("========= Q2 ==========");
        Pair<Integer, Integer> pair = cons(3, 4);
        int carResult = car(pair);
        int cdrResult = cdr(pair);

        System.out.println("car(cons(3, 4)) returns: " + carResult);
        System.out.println("cdr(cons(3, 4)) returns: " + cdrResult);

        /*
         * Q3.
         * Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the
         * number of ways it can be decoded.
         * For example, the message '111' would give 3, since it could be decoded as
         * 'aaa', 'ka', and 'ak'.
         * You can assume that the messages are decodable. For example, '001' is not
         * allowed.
         */
        System.out.println("========= Q3 ==========");
        String message = "111";
        int numWays = numDecodings(message);
        System.out.println("Number of ways to decode '" + message + "': " + numWays);

        /*
         * Q4.
         * Implement a job scheduler which takes in a function f and an integer n, and
         * calls f after n milliseconds.
         */
        Runnable myTask = () -> System.out.println("Job executed after delay");

        long delayMillis = 3000;

        scheduleJob(myTask, delayMillis);
        System.out.println("Job scheduled");

        /*
         * Q5.
         * Implement an autocomplete system. That is, given a query string s and a set
         * of all possible query strings, return all strings in the set that have s as a
         * prefix.
         * For example, given the query string de and the set of strings [dog, deer,
         * deal], return [deer, deal].
         * Hint: Try preprocessing the dictionary into a more efficient data structure
         * to speed up queries.
         */
        System.out.println("========= Q5 ==========");
        AutocompleteSystem autocomplete = new AutocompleteSystem();
        autocomplete.insert("dog");
        autocomplete.insert("deer");
        autocomplete.insert("deal");

        String query = "de";
        List<String> autocompleteResults = autocomplete.search(query);

        System.out.println("Autocomplete results for '" + query + "': " + autocompleteResults);

        /*
         * Q6.
         * The area of a circle is defined as πr^2. Estimate π to 3 decimal places using
         * a Monte Carlo method.
         * Hint: The basic equation of a circle is x2 + y2 = r2.
         */
        System.out.println("========= Q6 ==========");

        double pi = estimatePieDecimalPlaces();
        System.out.printf("Estimated value of π: %.3f%n", pi);

        /*
         * Q7.
         * Given a stream of elements too large to store in memory, pick a random
         * element from the stream with uniform probability.
         */
        System.out.println("========= Q7 ==========");
        RandomElementPicker<Integer> picker = new RandomElementPicker<>();

        for (int i = 0; i < 1000; i++) {
            picker.pickElement(i);
        }

        Integer randomElement = picker.getRandomElement();
        System.out.println("Random element: " + randomElement);

        /*
         * Q8.
         * A builder is looking to build a row of N houses that can be of K different
         * colors. He has a goal of minimizing cost while ensuring that no two
         * neighboring houses are of the same color.
         * Given an N by K matrix where the nth row and kth column represents the cost
         * to build the nth house with kth color, return the minimum cost which achieves
         * this goal.
         */
        System.out.println("========= Q8 ==========");
        int[][] costs = {
                { 1, 2, 3 }, // Cost of painting the first house with different colors
                { 4, 5, 6 }, // Cost of painting the second house with different colors
                { 7, 8, 9 } // Cost of painting the third house with different colors
        };

        int minCost = minCost(costs);

        System.out.println("Minimum cost: " + minCost);

        /*
         * Q9.
         * Given a dictionary of words and a string made up of those words (no spaces),
         * return the original sentence in a list. If there is more than one possible
         * reconstruction, return any of them. If there is no possible reconstruction,
         * then return null.
         * For example, given the set of words 'quick', 'brown', 'the', 'fox', and the
         * string "thequickbrownfox", you should return ['the', 'quick', 'brown',
         * 'fox'].
         * Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the
         * string "bedbathandbeyond", return either ['bed', 'bath', 'and', 'beyond] or
         * ['bedbath', 'and', 'beyond'].
         */
        System.out.println("========= Q9 ==========");
        Set<String> wordDict1 = new HashSet<>(Arrays.asList("quick", "brown", "the", "fox"));
        String sentence1 = "thequickbrownfox";
        List<String> result1 = reconstructSentence(wordDict1, sentence1);
        System.out.println("Reconstructed Sentence 1: " + result1);

        Set<String> wordDict2 = new HashSet<>(Arrays.asList("bed", "bath", "bedbath", "and", "beyond"));
        String sentence2 = "bedbathandbeyond";
        List<String> result2 = reconstructSentence(wordDict2, sentence2);
        System.out.println("Reconstructed Sentence 2: " + result2);

        /*
         * Q10.
         * Implement locking in a binary tree. A binary tree node can be locked or
         * unlocked only if all of its descendants or ancestors are not locked.
         * Design a binary tree node class with the following methods:
         * is_locked, which returns whether the node is locked
         * lock, which attempts to lock the node. If it cannot be locked, then it should
         * return false. Otherwise, it should lock it and return true.
         * unlock, which unlocks the node. If it cannot be unlocked, then it should
         * return false. Otherwise, it should unlock it and return true.
         * You may augment the node to add parent pointers or any other property you
         * would like. You may assume the class is used in a single-threaded program, so
         * there is no need for actual locks or mutexes. Each method should run in O(h),
         * where h is the height of the tree.
         */
        System.out.println("========= Q10 ==========");
        LockingTreeNode root = new LockingTreeNode();
        LockingTreeNode node1 = new LockingTreeNode();
        LockingTreeNode node2 = new LockingTreeNode();
        LockingTreeNode node3 = new LockingTreeNode();
        LockingTreeNode node4 = new LockingTreeNode();

        root.left = node1;
        root.right = node2;
        node1.parent = root;
        node2.parent = root;
        node1.left = node3;
        node3.parent = node1;
        node2.right = node4;
        node4.parent = node2;

        System.out.println("Locking node1: " + node1.lock()); // true
        System.out.println("Is node1 locked? " + node1.isLocked()); // true

        System.out.println("Locking node3: " + node3.lock()); // false
        System.out.println("Is node3 locked? " + node3.isLocked()); // false

        System.out.println("Unlocking node1: " + node1.unlock()); // true
        System.out.println("Is node1 locked? " + node1.isLocked()); // false

        System.out.println("Locking node2: " + node2.lock()); // true
        System.out.println("Is node2 locked? " + node2.isLocked()); // true

        System.out.println("Locking node1: " + node1.lock()); // true
        System.out.println("Is node1 locked? " + node1.isLocked()); // true

        System.out.println("Unlocking node2: " + node2.unlock()); // true
        System.out.println("Is node2 locked? " + node2.isLocked()); // false

    }
}
