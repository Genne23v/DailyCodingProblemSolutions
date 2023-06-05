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
import java.util.PriorityQueue;
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

    // S11.
    static class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
        }
    }

    public static ListNode removeKthLast(ListNode head, int k) {
        ListNode fast = head;
        ListNode slow = head;

        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }

        // If fast is null, it means k is equal to the length of the list
        // So we need to remove the head of the list
        if (fast == null) {
            return head.next;
        }

        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return head;
    }

    // S12.
    public static List<String> justifyText(String[] words, int k) {
        List<String> justifiedLines = new ArrayList<>();
        int n = words.length;
        int i = 0;

        while (i < n) {
            StringBuilder line = new StringBuilder();
            int lineLength = words[i].length();
            int wordCount = 1;
            int j = i + 1;

            while (j < n && lineLength + 1 + words[j].length() <= k) {
                lineLength += 1 + words[j].length();
                wordCount++;
                j++;
            }

            int extraSpaces = k - lineLength;
            int spacesPerWord = wordCount > 1 ? extraSpaces / (wordCount - 1) : extraSpaces;

            line.append(words[i]);

            for (int x = i + 1; x < j; x++) {
                for (int s = 0; s < spacesPerWord; s++) {
                    line.append(" ");
                }

                if (x - i <= extraSpaces % (wordCount - 1)) {
                    line.append(" ");
                }

                line.append(" ").append(words[x]);
            }

            if (wordCount == 1) {
                while (line.length() < k) {
                    line.append(" ");
                }
            }

            justifiedLines.add(line.toString());

            i = j;
        }

        return justifiedLines;
    }

    // S13.
    public static int trapWater(int[] heights) {
        int left = 0;
        int right = heights.length - 1;
        int maxLeft = 0;
        int maxRight = 0;
        int totalWater = 0;

        while (left <= right) {
            if (heights[left] <= heights[right]) {
                if (heights[left] > maxLeft) {
                    maxLeft = heights[left];
                } else {
                    totalWater += maxLeft - heights[left];
                }
                left++;
            } else {
                if (heights[right] > maxRight) {
                    maxRight = heights[right];
                } else {
                    totalWater += maxRight - heights[right];
                }
                right--;
            }
        }

        return totalWater;
    }

    // S14.
    public static String findPalindrome(String s) {
        int n = s.length();
        int[][] dp = new int[n + 1][n + 1];

        for (int i = 1; i <= n; i++) {
            dp[i][i] = 0;
        }

        // the minimum number of insertions required to make the substring from index i
        // to j a palindrome.
        for (int len = 2; len <= n; len++) {
            for (int i = 1; i <= n - len + 1; i++) {
                int j = i + len - 1;

                if (s.charAt(i - 1) == s.charAt(j - 1)) { // Insertions are not needed
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i][j - 1], dp[i + 1][j]) + 1;
                }
            }
        }

        StringBuilder palindrome = new StringBuilder();
        int i = 1;
        int j = n;

        while (i < j) {
            if (s.charAt(i - 1) == s.charAt(j - 1)) {
                palindrome.append(s.charAt(i - 1));
                i++;
                j--;
            } else if (dp[i][j - 1] <= dp[i + 1][j]) { // If left is smaller, add a ch at j
                palindrome.append(s.charAt(j - 1));
                j--;
            } else { // If down is smaller, add a ch at i
                palindrome.append(s.charAt(i - 1));
                i++;
            }
        }

        int endIndexToCopy = palindrome.length() - 1;
        if (i == j) {
            palindrome.append(s.charAt(i - 1));
        }

        for (i = endIndexToCopy; i >= 0; i--) {
            palindrome.append(palindrome.charAt(i));
        }

        return palindrome.toString();
    }

    // S15.
    static class TreeNode<T> {
        T val;
        TreeNode<T> left;
        TreeNode<T> right;

        TreeNode(T val) {
            this.val = val;
        }
    }

    public static int findSecondLargest(TreeNode<Integer> root) {
        if (root == null)
            throw new IllegalArgumentException("Invalid input: BST is empty");

        TreeNode<Integer> largest = findLargest(root);
        if (largest.left != null)
            return findLargest(largest.left).val;
        else
            return findParentOfLargest(root, largest).val;
    }

    private static TreeNode<Integer> findLargest(TreeNode<Integer> node) {
        while (node.right != null) {
            node = node.right;
        }
        return node;
    }

    private static TreeNode<Integer> findParentOfLargest(TreeNode<Integer> root, TreeNode<Integer> largest) {
        TreeNode<Integer> parent = null;
        TreeNode<Integer> current = root;

        while (current != null && current != largest) {
            parent = current;
            current = current.right;
        }

        return parent;
    }

    // S16.
    static class Cell {
        private int x;
        private int y;
        private boolean alive;

        public Cell(int x, int y, boolean alive) {
            this.x = x;
            this.y = y;
            this.alive = alive;
        }

        public int getX() {
            return x;
        }

        public int getY() {
            return y;
        }

        public boolean isAlive() {
            return alive;
        }

        public void setAlive(boolean alive) {
            this.alive = alive;
        }
    }

    static class GameOfLife {
        private Cell[][] board;

        public GameOfLife(int size) {
            board = new Cell[size][size];

            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    board[i][j] = new Cell(i, j, false);
                }
            }

        }

        public void initialize(int[][] liveCellCoordinates) {
            for (int[] coordinate : liveCellCoordinates) {
                int x = coordinate[0];
                int y = coordinate[1];
                board[x][y] = new Cell(x, y, true);
            }
        }

        public void run(int steps) {
            for (int step = 1; step <= steps; step++) {
                System.out.println("Step " + step + ":");
                printBoard();

                Cell[][] nextBoard = new Cell[board.length][board.length];

                for (int i = 0; i < board.length; i++) {
                    for (int j = 0; j < board.length; j++) {
                        nextBoard[i][j] = getNextCellState(i, j);
                    }
                }

                board = nextBoard;
            }
        }

        private Cell getNextCellState(int x, int y) {
            int liveNeighbors = countLiveNeighbors(x, y);
            Cell currentCell = board[x][y];
            Cell nextCell = new Cell(x, y, currentCell.isAlive());

            if (currentCell.isAlive()) {
                if (liveNeighbors < 2 || liveNeighbors > 3) {
                    nextCell.setAlive(false);
                }
            } else {
                if (liveNeighbors == 3) {
                    nextCell.setAlive(true);
                }
            }

            return nextCell;
        }

        private int countLiveNeighbors(int x, int y) {
            int count = 0;

            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    if (i == x && j == y) {
                        continue; // Skip the current cell
                    }

                    if (isValidCoordinate(i, j) && board[i][j] != null && board[i][j].isAlive()) {
                        count++;
                    }
                }
            }

            return count;
        }

        private boolean isValidCoordinate(int x, int y) {
            return x >= 0 && x < board.length && y >= 0 && y < board.length;
        }

        private void printBoard() {
            int minX = Integer.MAX_VALUE;
            int maxX = Integer.MIN_VALUE;
            int minY = Integer.MAX_VALUE;
            int maxY = Integer.MIN_VALUE;

            for (Cell[] row : board) {
                for (Cell cell : row) {
                    if (cell != null && cell.isAlive()) {
                        minX = Math.min(minX, cell.getX());
                        maxX = Math.max(maxX, cell.getX());
                        minY = Math.min(minY, cell.getY());
                        maxY = Math.max(maxY, cell.getY());
                    }
                }
            }

            for (int i = minX; i <= maxX; i++) {
                for (int j = minY; j <= maxY; j++) {
                    Cell cell = board[i][j];
                    char symbol = (cell != null && cell.isAlive()) ? '*' : '.';
                    System.out.print(symbol);
                }
                System.out.println();
            }

            System.out.println();
        }
    }

    // S17.
    public static List<String> findItinerary(List<String[]> flights, String startAirport) {
        Map<String, PriorityQueue<String>> flightMap = new HashMap<>();

        for (String[] flight : flights) {
            String origin = flight[0];
            String destination = flight[1];
            flightMap.putIfAbsent(origin, new PriorityQueue<>());
            flightMap.get(origin).offer(destination);
        }

        List<String> itinerary = new ArrayList<>();
        dfs(flightMap, startAirport, itinerary);

        if (itinerary.size() != flights.size() + 1) {
            return null;
        }

        return itinerary;
    }

    private static void dfs(Map<String, PriorityQueue<String>> flightMap, String airport, List<String> itinerary) {
        PriorityQueue<String> destinations = flightMap.get(airport);
        while (destinations != null && !destinations.isEmpty()) {
            String nextAirport = destinations.poll();
            dfs(flightMap, nextAirport, itinerary);
        }
        itinerary.add(0, airport);
    }

    // S18.
    public static int countInversions(int[] arr) {
        if (arr == null || arr.length <= 1) {
            return 0;
        }

        int[] temp = new int[arr.length];
        return mergeSortAndCount(arr, temp, 0, arr.length - 1);
    }

    private static int mergeSortAndCount(int[] arr, int[] temp, int left, int right) {
        int count = 0;
        if (left < right) {
            int mid = left + (right - left) / 2;
            count += mergeSortAndCount(arr, temp, left, mid);
            count += mergeSortAndCount(arr, temp, mid + 1, right);
            count += merge(arr, temp, left, mid + 1, right);
        }
        return count;
    }

    private static int merge(int[] arr, int[] temp, int left, int mid, int right) {
        int i = left;
        int j = mid;
        int k = left;
        int count = 0;

        // Compare both halves and merge them in sorted order
        while (i <= mid - 1 && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
                count += mid - i; // Count the number of inversions
            }
        }

        // Copy the remaining elements of left array
        while (i <= mid - 1) {
            temp[k++] = arr[i++];
        }

        while (j <= right) {
            temp[k++] = arr[j++];
        }

        System.arraycopy(temp, left, arr, left, right - left + 1);
        return count;
    }

    // S19.
    public static TreeNode<Character> buildTree(char[] preorder, char[] inorder) {
        if (preorder == null || inorder == null || preorder.length != inorder.length) {
            return null;
        }

        return buildTreeHelper(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }

    private static TreeNode<Character> buildTreeHelper(char[] preorder, int preStart, int preEnd, char[] inorder,
            int inStart,
            int inEnd) {
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }

        char rootVal = preorder[preStart];
        TreeNode<Character> root = new TreeNode<>(rootVal);

        int rootIndexInorder = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == rootVal) {
                rootIndexInorder = i;
                break;
            }
        }

        int leftSubtreeSize = rootIndexInorder - inStart;
        root.left = buildTreeHelper(preorder, preStart + 1, preStart + leftSubtreeSize, inorder, inStart,
                rootIndexInorder - 1);
        root.right = buildTreeHelper(preorder, preStart + leftSubtreeSize + 1, preEnd, inorder, rootIndexInorder + 1,
                inEnd);

        return root;
    }

    public static void printInorder(TreeNode<Character> root) {
        if (root == null) {
            return;
        }

        printInorder(root.left);
        System.out.print(root.val + " ");
        printInorder(root.right);
    }

    // S20.
    public static int findMaxSubarraySum(int[] nums) {
        int maxSum = 0;
        int currentSum = 0;

        for (int num : nums) {
            currentSum = Math.max(num, currentSum + num);
            maxSum = Math.max(maxSum, currentSum);
        }

        return maxSum;
    }

    // S21.

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

        /*
         * Q11.
         * Given a singly linked list and an integer k, remove the kth last element from
         * the list. k is guaranteed to be smaller than the length of the list.
         * The list is very long, so making more than one pass is prohibitively
         * expensive.
         * Do this in constant space and in one pass.
         */
        System.out.println("========= Q11 ==========");
        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(3);
        head.next.next.next = new ListNode(4);
        head.next.next.next.next = new ListNode(5);

        int k = 2;
        ListNode result = removeKthLast(head, k);

        while (result != null) {
            System.out.print(result.val + " ");
            result = result.next;
        }

        /*
         * Q12.
         * Write an algorithm to justify text. Given a sequence of words and an integer
         * line length k, return a list of strings which represents each line, fully
         * justified.
         * More specifically, you should have as many words as possible in each line.
         * There should be at least one space between each word. Pad extra spaces when
         * necessary so that each line has exactly length k. Spaces should be
         * distributed as equally as possible, with the extra spaces, if any,
         * distributed starting from the left.
         * If you can only fit one word on a line, then you should pad the right-hand
         * side with spaces.
         * Each word is guaranteed not to be longer than k.
         * For example, given the list of words ["the", "quick", "brown", "fox",
         * "jumps", "over", "the", "lazy", "dog"] and k = 16, you should return the
         * following:
         * ["the  quick brown", # 1 extra space on the left
         * "fox  jumps  over", # 2 extra spaces distributed evenly
         * "the   lazy   dog"] # 4 extra spaces distributed evenly
         */
        System.out.println("========= Q12 ==========");
        String[] words = { "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog" };
        int lineLength = 16;

        List<String> justifiedLines = justifyText(words, lineLength);

        for (String line : justifiedLines) {
            System.out.println(line);
        }

        /*
         * Q13.
         * You are given an array of non-negative integers that represents a
         * two-dimensional elevation map where each element is unit-width wall and the
         * integer is the height. Suppose it will rain and all spots between two walls
         * get filled up.
         * Compute how many units of water remain trapped on the map in O(N) time and
         * O(1) space.
         * For example, given the input [2, 1, 2], we can hold 1 unit of water in the
         * middle.
         * Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2
         * in the second, and 3 in the fourth index (we cannot hold 5 since it would run
         * off to the left), so we can trap 8 units of water.
         */
        System.out.println("========= Q13 ==========");
        int[] heights1 = { 2, 1, 2 };
        System.out.println(trapWater(heights1)); // Output: 1

        int[] heights2 = { 3, 0, 1, 3, 0, 5 };
        System.out.println(trapWater(heights2)); // Output: 8

        /*
         * Q14.
         * Given a string, find the palindrome that can be made by inserting the fewest
         * number of characters as possible anywhere in the word. If there is more than
         * one palindrome of minimum length that can be made, return the
         * lexicographically earliest one (the first one alphabetically).
         * For example, given the string "race", you should return "ecarace", since we
         * can add three letters to it (which is the smallest amount to make a
         * palindrome). There are seven other palindromes that can be made from "race"
         * by adding three letters, but "ecarace" comes first alphabetically.
         * As another example, given the string "google", you should return "elgoogle".
         */
        System.out.println("========= Q14 ==========");
        String input1 = "race";
        System.out.println(findPalindrome(input1)); // Output: ecarace

        String input2 = "google";
        System.out.println(findPalindrome(input2)); // Output: elgoogle

        /*
         * Q15.
         * Given the root to a binary search tree, find the second largest node in the
         * tree.
         */
        System.out.println("========= Q15 ==========");
        TreeNode binaryTreeRoot = new TreeNode(6);
        binaryTreeRoot.left = new TreeNode(2);
        binaryTreeRoot.right = new TreeNode(8);
        binaryTreeRoot.left.left = new TreeNode(1);
        binaryTreeRoot.left.right = new TreeNode(4);
        binaryTreeRoot.right.left = new TreeNode(7);
        binaryTreeRoot.right.right = new TreeNode(9);

        int secondLargest = findSecondLargest(binaryTreeRoot);
        System.out.println("Second largest node: " + secondLargest); // Output: 8

        /*
         * Q16.
         * Conway's Game of Life takes place on an infinite two-dimensional board of
         * square cells. Each cell is either dead or alive, and at each tick, the
         * following rules apply:
         * Any live cell with less than two live neighbours dies.
         * Any live cell with two or three live neighbours remains living.
         * Any live cell with more than three live neighbours dies.
         * Any dead cell with exactly three live neighbours becomes a live cell.
         * A cell neighbours another cell if it is horizontally, vertically, or
         * diagonally adjacent.
         * Implement Conway's Game of Life. It should be able to be initialized with a
         * starting list of live cell coordinates and the number of steps it should run
         * for. Once initialized, it should print out the board state at each step.
         * Since it's an infinite board, print out only the relevant coordinates, i.e.
         * from the top-leftmost live cell to bottom-rightmost live cell.
         * You can represent a live cell with an asterisk (*) and a dead cell with a dot
         * (.).
         */
        System.out.println("========= Q16 ==========");
        int[][] liveCellCoordinates = {
                { 1, 2 },
                { 2, 2 },
                { 2, 3 },
                { 3, 1 },
                { 3, 2 }
        };

        int size = 5;
        int steps = 5;

        GameOfLife game = new GameOfLife(size);
        game.initialize(liveCellCoordinates);
        game.run(steps);

        /*
         * Q17.
         * Given an unordered list of flights taken by someone, each represented as
         * (origin, destination) pairs, and a starting airport, compute the person's
         * itinerary. If no such itinerary exists, return null. If there are multiple
         * possible itineraries, return the lexicographically smallest one. All flights
         * must be used in the itinerary.
         * For example, given the list of flights [('SFO', 'HKO'), ('YYZ', 'SFO'),
         * ('YUL', 'YYZ'), ('HKO', 'ORD')] and starting airport 'YUL', you should return
         * the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].
         * Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting
         * airport 'COM', you should return null.
         * Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
         * and starting airport 'A', you should return the list ['A', 'B', 'C', 'A',
         * 'C'] even though ['A', 'C', 'A', 'B', 'C'] is also a valid itinerary.
         * However, the first one is lexicographically smaller.
         */
        System.out.println("========= Q17 ==========");
        List<String[]> flights1 = Arrays.asList(
                new String[] { "SFO", "HKO" },
                new String[] { "YYZ", "SFO" },
                new String[] { "YUL", "YYZ" },
                new String[] { "HKO", "ORD" });
        String startAirport1 = "YUL";
        List<String> itineraryResult1 = findItinerary(flights1, startAirport1);
        System.out.println("Itinerary 1: " + itineraryResult1);

        List<String[]> flights2 = Arrays.asList(
                new String[] { "SFO", "COM" },
                new String[] { "COM", "YYZ" });
        String startAirport2 = "COM";
        List<String> itineraryResult2 = findItinerary(flights2, startAirport2);
        System.out.println("Itinerary 2: " + itineraryResult2);

        List<String[]> flights3 = Arrays.asList(
                new String[] { "A", "B" },
                new String[] { "A", "C" },
                new String[] { "B", "C" },
                new String[] { "C", "A" });
        String startAirport3 = "A";
        List<String> itineraryResult3 = findItinerary(flights3, startAirport3);
        System.out.println("Itinerary 3: " + itineraryResult3);

        /*
         * Q18.
         * We can determine how "out of order" an array A is by counting the number of
         * inversions it has. Two elements A[i] and A[j] form an inversion if A[i] >
         * A[j] but i < j. That is, a smaller element appears after a larger element.
         * Given an array, count the number of inversions it has. Do this faster than
         * O(N^2) time.
         * You may assume each element in the array is distinct.
         * For example, a sorted list has zero inversions. The array [2, 4, 1, 3, 5] has
         * three inversions: (2, 1), (4, 1), and (4, 3). The array [5, 4, 3, 2, 1] has
         * ten inversions: every distinct pair forms an inversion.
         */
        System.out.println("========= Q18 ==========");
        int[] arr1 = { 2, 4, 1, 3, 5 };
        int inversions1 = countInversions(arr1);
        System.out.println("Inversions in arr1: " + inversions1);

        int[] arr2 = { 5, 4, 3, 2, 1 };
        int inversions2 = countInversions(arr2);
        System.out.println("Inversions in arr2: " + inversions2);

        /*
         * Q19.
         * Given pre-order and in-order traversals of a binary tree, write a function to
         * reconstruct the tree.
         * For example, given the following preorder traversal:
         * [a, b, d, e, c, f, g]
         * And the following inorder traversal:
         * [d, b, e, a, f, c, g]
         * You should return the following tree:
         * "    a      "
         * "   / \     "
         * "  b   c    "
         * " / \ / \   "
         * "d  e f  g  "
         */
        System.out.println("========= Q19 ==========");
        char[] preorder = { 'a', 'b', 'd', 'e', 'c', 'f', 'g' };
        char[] inorder = { 'd', 'b', 'e', 'a', 'f', 'c', 'g' };

        TreeNode<Character> constructedTreeRoot = buildTree(preorder, inorder);

        System.out.println("In-order traversal of the reconstructed tree:");
        printInorder(constructedTreeRoot);
        System.out.println();

        /*
         * Q20.
         * Given an array of numbers, find the maximum sum of any contiguous subarray of
         * the array.
         * For example, given the array [34, -50, 42, 14, -5, 86], the maximum sum would
         * be 137, since we would take elements 42, 14, -5, and 86.
         * Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would
         * not take any elements.
         * Do this in O(N) time.
         */
        System.out.println("========= Q20 ==========");
        int[] arrToFindMaxSubarray1 = { 34, -50, 42, 14, -5, 86 };
        int[] arrToFindMaxSubarray2 = { -5, -1, -8, -9 };

        int maxSum1 = findMaxSubarraySum(arrToFindMaxSubarray1);
        int maxSum2 = findMaxSubarraySum(arrToFindMaxSubarray2);

        System.out.println("Maximum sum in arr1: " + maxSum1); // Output: 137
        System.out.println("Maximum sum in arr2: " + maxSum2); // Output: 0

        /*
         * Q21.
         * Given a function that generates perfectly random numbers between 1 and k
         * (inclusive), where k is an input, write a function that shuffles a deck of
         * cards represented as an array using only swaps.
         * It should run in O(N) time.
         * Hint: Make sure each one of the 52! permutations of the deck is equally
         * likely.
         */
        System.out.println("========= Q21 ==========");

    }
}
