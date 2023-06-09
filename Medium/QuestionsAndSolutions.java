package Medium;

import java.util.Random;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.NoSuchElementException;

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

        public Node(T item) {
            this.val = item;
            this.left = this.right = null;
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
    public static void shuffleDeck(int[] deck) {
        Random rand = new Random();
        int n = deck.length;

        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            swap(deck, i, j);
        }
    }

    private static void swap(int[] deck, int i, int j) {
        int temp = deck[i];
        deck[i] = deck[j];
        deck[j] = temp;
    }

    // S22.
    static class QueueUsingStacks<T> {
        private Stack<T> enqueueStack;
        private Stack<T> dequeueStack;

        public QueueUsingStacks() {
            enqueueStack = new Stack<>();
            dequeueStack = new Stack<>();
        }

        public void enqueue(T element) {
            enqueueStack.push(element);
        }

        public T dequeue() {
            if (isEmpty()) {
                throw new IllegalStateException("Queue is empty");
            }

            if (dequeueStack.isEmpty()) {
                // Transfer elements from enqueueStack to dequeueStack in reverse order
                while (!enqueueStack.isEmpty()) {
                    dequeueStack.push(enqueueStack.pop());
                }
            }

            return dequeueStack.pop();
        }

        public boolean isEmpty() {
            return enqueueStack.isEmpty() && dequeueStack.isEmpty();
        }

        public int size() {
            return enqueueStack.size() + dequeueStack.size();
        }
    }

    // S23.
    static class GraphColoring {
        private int[][] graph;
        private int numVertices;
        private int[] colors;

        public GraphColoring(int[][] graph) {
            this.graph = graph;
            this.numVertices = graph.length;
            this.colors = new int[numVertices];
        }

        public boolean canColorGraph(int k) {
            return canColorVertex(0, k);
        }

        private boolean canColorVertex(int vertex, int k) {
            if (vertex == numVertices) {
                return true; // All vertices have been colored
            }

            for (int color = 1; color <= k; color++) {
                if (isColorValid(vertex, color)) {
                    colors[vertex] = color;

                    if (canColorVertex(vertex + 1, k)) {
                        return true;
                    }

                    colors[vertex] = 0; // Backtrack and try a different color
                }
            }

            return false;
        }

        private boolean isColorValid(int vertex, int color) {
            for (int i = 0; i < numVertices; i++) {
                if (graph[vertex][i] == 1 && colors[i] == color) {
                    return false;
                }
            }

            return true;
        }
    }

    // S24.
    public static List<String> breakLines(String s, int k) {
        String[] words = s.split(" ");
        List<String> lines = new ArrayList<>();
        StringBuilder currentLine = new StringBuilder();
        int currentLength = 0;

        for (String word : words) {
            int wordLength = word.length();

            if (currentLength + wordLength <= k) {
                currentLine.append(word).append(" ");
                currentLength += wordLength + 1;
            } else {
                lines.add(currentLine.toString().trim());
                currentLine = new StringBuilder(word).append(" ");
                currentLength = wordLength + 1;
            }
        }

        lines.add(currentLine.toString().trim());

        return lines;
    }

    // S25.
    public static Integer search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            }

            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        return null;
    }

    // S26.
    public static boolean canPartition(int[] nums) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }

        if (totalSum % 2 != 0) {
            return false; // If the total sum is odd, it cannot be partitioned into two equal subsets
        }

        int targetSum = totalSum / 2;
        int n = nums.length;
        boolean[][] dp = new boolean[n + 1][targetSum + 1];

        // Initialize the first column with true, as we can make a sum of 0 with an
        // empty subset
        for (int i = 0; i <= n; i++) {
            dp[i][0] = true;
        }

        // Fill the dp array using the subset sum bottom-up approach
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= targetSum; j++) {
                if (j < nums[i - 1]) {
                    dp[i][j] = dp[i - 1][j]; // Copy previous number's result
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]]; // Copy previous number's result or use the
                                                                           // result without previous number
                }
            }
        }

        return dp[n][targetSum];
    }

    // S27.
    public static long pow(int x, int y) {
        if (y < 0) {
            return pow(1 / x, -y);
        } else if (y == 0) {
            return 1;
        } else if (y == 1) {
            return x;
        } else if (y % 2 == 0) {
            long halfPower = pow(x, y / 2);
            return halfPower * halfPower;
        } else {
            long halfPower = pow(x, y / 2);
            return x * halfPower * halfPower;
        }
    }

    // S28.
    public static int countWays(int N, int M) {
        int[][] dp = new int[N][M];

        for (int i = 0; i < N; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < M; j++) {
            dp[0][j] = 1;
        }

        for (int i = 1; i < N; i++) {
            for (int j = 1; j < M; j++) {
                // The number of ways to reach cell (i, j) is the sum of the ways
                // to reach the cell above (i-1, j) and the cell to the left (i, j-1)
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[N - 1][M - 1];
    }

    // S29.
    public static int tossUnbiased() {
        while (true) {
            int toss1 = tossBiased();
            int toss2 = tossBiased();

            if (toss1 != toss2) {
                return toss1;
            }
        }
    }

    // Assume this is the provided biased coin toss function
    public static int tossBiased() {
        double probabilityOfHeads = 0.3;

        double random = Math.random();

        if (random < probabilityOfHeads) {
            return 1;
        } else {
            return 0;
        }
    }

    // S30.
    public static int countAttackingPairs(int[][] bishops, int M) {
        Map<Integer, Integer> positiveSlopes = new HashMap<>();
        Map<Integer, Integer> negativeSlopes = new HashMap<>();

        int pairs = 0;

        for (int[] bishop : bishops) {
            int row = bishop[0];
            int col = bishop[1];

            // Calculate the positive and negative diagonal slopes
            int positiveSlope = row + col;
            int negativeSlope = row - col;

            positiveSlopes.put(positiveSlope, positiveSlopes.getOrDefault(positiveSlope, 0) + 1);
            negativeSlopes.put(negativeSlope, negativeSlopes.getOrDefault(negativeSlope, 0) + 1);
        }

        for (int count : positiveSlopes.values()) {
            pairs += countPairs(count);
        }
        for (int count : negativeSlopes.values()) {
            pairs += countPairs(count);
        }

        return pairs;
    }

    private static int countPairs(int count) {
        // Calculate the number of pairs using combination formula (nC2)
        return count * (count - 1) / 2;
    }

    // S31.
    public static int countOccurrences(int N, int X) {
        int count = 0;

        for (int i = 1; i <= N; i++) {
            if (X % i == 0 && X / i <= N) {
                count++;
            }
        }

        return count;
    }

    // S32.
    public static int minColumnRemovals(String[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int rowCount = matrix.length;
        int colCount = matrix[0].length;
        int removalCount = 0;

        for (int col = 0; col < colCount; col++) {
            for (int row = 1; row < rowCount; row++) {
                if (matrix[row][col].compareTo(matrix[row - 1][col]) < 0) {
                    removalCount++;
                    break;
                }
            }
        }

        return removalCount;
    }

    // S33.
    public static ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }

        PriorityQueue<ListNode> queue = new PriorityQueue<>((a, b) -> a.val - b.val);

        for (ListNode head : lists) {
            if (head != null) {
                queue.offer(head);
            }
        }

        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;

        while (!queue.isEmpty()) {
            ListNode node = queue.poll();
            curr.next = node;
            curr = curr.next;

            if (node.next != null) {
                queue.offer(node.next);
            }
        }

        return dummy.next;
    }

    public static void printList(ListNode head) {
        ListNode curr = head;
        while (curr != null) {
            System.out.print(curr.val + " ");
            curr = curr.next;
        }
        System.out.println();
    }

    // S34.
    public static boolean checkPossibility(int[] nums) {
        int count = 0;

        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] > nums[i + 1]) {
                count++;
                if (count > 1) {
                    return false;
                }
                // Check if we can modify the current element or the next element
                if (i > 0 && nums[i - 1] > nums[i + 1]) {
                    nums[i + 1] = nums[i];
                } else {
                    nums[i] = nums[i + 1];
                }
            }
        }

        return true;
    }

    // S35.
    public static <T> TreeNode<T> invertTree(TreeNode<T> root) {
        if (root == null) {
            return null;
        }

        // Swap the left and right children of the current node
        TreeNode<T> temp = root.left;
        root.left = root.right;
        root.right = temp;

        invertTree(root.left);
        invertTree(root.right);

        return root;
    }

    public static <T> void printTree(TreeNode<T> root) {
        if (root == null) {
            return;
        }

        System.out.println(root.val);
        printTree(root.left);
        printTree(root.right);
    }

    // S36.
    public static int countIslands(int[][] matrix) {
        int count = 0;
        int rows = matrix.length;
        int cols = matrix[0].length;
        boolean[][] visited = new boolean[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 1 && !visited[i][j]) {
                    exploreIsland(matrix, visited, i, j);
                    count++;
                }
            }
        }

        return count;
    }

    private static void exploreIsland(int[][] matrix, boolean[][] visited, int row, int col) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        if (row < 0 || col < 0 || row >= rows || col >= cols || matrix[row][col] == 0 || visited[row][col]) {
            return;
        }

        visited[row][col] = true;

        exploreIsland(matrix, visited, row - 1, col);
        exploreIsland(matrix, visited, row + 1, col);
        exploreIsland(matrix, visited, row, col - 1);
        exploreIsland(matrix, visited, row, col + 1);
    }

    // S37.
    public static int select(int x, int y, int b) {
        // If b = 1, all bits will be 1 and return x
        // If b = 0, change all bits to 1 to return y
        return (x & -b) | (y & -(b ^ 1));
    }

    // S38.
    public static int minRemoval(String s) {
        int count = 0;
        Stack<Character> stack = new Stack<>();

        for (char c : s.toCharArray()) {
            if (c == '(') {
                stack.push(c);
            } else if (c == ')') {
                if (!stack.isEmpty() && stack.peek() == '(') {
                    stack.pop();
                } else {
                    count++;
                }
            }
        }

        count += stack.size(); // Add the remaining opening parentheses in the stack

        return count;
    }

    // S39.
    public static int divide(int dividend, int divisor) {
        if (divisor == 0) {
            throw new ArithmeticException("Divisor cannot be zero.");
        }

        if (dividend == 0) {
            return 0;
        }

        if (dividend < divisor) {
            return 0;
        }

        // Perform repeated subtraction
        int quotient = 0;
        while (dividend >= divisor) {
            dividend -= divisor;
            quotient++;
        }

        return quotient;
    }

    // S40.
    static class BinarySearchTree {
        Node<Integer> root;

        BinarySearchTree() {
            root = null;
        }

        void insert(int key) {
            root = insertRec(root, key);
        }

        Node<Integer> insertRec(Node<Integer> root, int key) {
            if (root == null) {
                root = new Node<>(key);
                return root;
            }

            if (key < root.val)
                root.left = insertRec(root.left, key);
            else if (key > root.val)
                root.right = insertRec(root.right, key);

            return root;
        }

        void delete(int key) {
            root = deleteRec(root, key);
        }

        Node<Integer> deleteRec(Node<Integer> root, int key) {
            if (root == null)
                return root;

            if (key < root.val)
                root.left = deleteRec(root.left, key);
            else if (key > root.val)
                root.right = deleteRec(root.right, key);
            else {
                if (root.left == null)
                    return root.right;
                else if (root.right == null)
                    return root.left;

                root.val = minValue(root.right);
                root.right = deleteRec(root.right, root.val);
            }

            return root;
        }

        int minValue(Node<Integer> root) {
            int minv = root.val;
            while (root.left != null) {
                minv = root.left.val;
                root = root.left;
            }
            return minv;
        }

        boolean search(int key) {
            return searchRec(root, key);
        }

        boolean searchRec(Node<Integer> root, int key) {
            if (root == null || root.val == key)
                return root != null;

            if (key < root.val)
                return searchRec(root.left, key);

            return searchRec(root.right, key);
        }

        void inorderTraversal() {
            inorderTraversalRec(root);
        }

        void inorderTraversalRec(Node<Integer> root) {
            if (root != null) {
                inorderTraversalRec(root.left);
                System.out.print(root.val + " ");
                inorderTraversalRec(root.right);
            }
        }
    }

    // S41.
    public static int generateRandomNumber(int n, List<Integer> l) {
        List<Integer> available = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (!l.contains(i)) {
                available.add(i);
            }
        }

        Random random = new Random();
        int index = random.nextInt(available.size());

        return available.get(index);
    }

    // S42.
    static class TimeMap {
        private Map<Integer, TreeMap<Integer, Integer>> map;

        public TimeMap() {
            map = new HashMap<>();
        }

        public void set(int key, int value, int time) {
            if (!map.containsKey(key)) {
                map.put(key, new TreeMap<>());
            }
            map.get(key).put(time, value);
        }

        public Integer get(int key, int time) {
            if (map.containsKey(key)) {
                TreeMap<Integer, Integer> values = map.get(key);
                Integer floorTime = values.floorKey(time);
                if (floorTime != null) {
                    return values.get(floorTime);
                }
            }
            return null;
        }
    }

    // S43.
    public static int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        int maxLength = 0;

        for (int num : nums) {
            set.add(num);
        }

        for (int num : nums) {
            if (!set.contains(num - 1)) {
                int currentNum = num;
                int currentLength = 1;

                while (set.contains(currentNum + 1)) {
                    currentNum++;
                    currentLength++;
                }

                maxLength = Math.max(maxLength, currentLength);
            }
        }
        return maxLength;
    }

    // S44.
    public static List<Integer> findContiguousElementsSum(int[] nums, int k) {
        List<Integer> result = new ArrayList<>();
        int left = 0;
        int right = 0;
        int sum = 0;

        while (right < nums.length) {
            sum += nums[right];

            while (sum > k) {
                sum -= nums[left];
                left++;
            }

            if (sum == k) {
                for (int i = left; i <= right; i++) {
                    result.add(nums[i]);
                }
                return result;
            }

            right++;
        }

        return result;
    }

    // S45.
    public static String shortestSubstring(String s, Set<Character> charSet) {
        Map<Character, Integer> charCounts = new HashMap<>();
        for (char c : charSet) {
            charCounts.put(c, charCounts.getOrDefault(c, 0) + 1);
        }

        int left = 0;
        int right = 0;
        int count = charSet.size();
        int minLen = Integer.MAX_VALUE;
        int minStart = 0;

        while (right < s.length()) {
            char c = s.charAt(right);
            if (charCounts.containsKey(c)) {
                charCounts.put(c, charCounts.get(c) - 1);
                if (charCounts.get(c) == 0) {
                    count--;
                }
            }

            while (count == 0) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }

                char leftChar = s.charAt(left);
                if (charCounts.containsKey(leftChar)) {
                    charCounts.put(leftChar, charCounts.get(leftChar) + 1);
                    if (charCounts.get(leftChar) > 0) {
                        count++;
                    }
                }

                left++;
            }

            right++;
        }

        if (minLen == Integer.MAX_VALUE) {
            return null;
        }

        return s.substring(minStart, minStart + minLen);
    }

    // S46.
    public static boolean canReachLastIndex(int[] nums) {
        int maxReach = 0;
        int n = nums.length;

        for (int i = 0; i < n; i++) {
            if (i > maxReach) {
                // If the current index is not reachable, return false
                return false;
            }
            maxReach = Math.max(maxReach, i + nums[i]);

            if (maxReach >= n - 1) {
                // If the maximum reachable index is greater than or equal to the last index,
                // we can reach the last index
                return true;
            }
        }
        return false;
    }

    // S47.
    public static int swapEvenOddBits(int n) {
        return ((n & 0xAA) >>> 1) | ((n & 0x55) << 1);
    }

    // S48.
    public static List<List<Integer>> binaryTreePaths(TreeNode<Integer> root) {
        List<List<Integer>> paths = new ArrayList<>();
        List<Integer> currentPath = new ArrayList<>();
        dfs(root, currentPath, paths);
        return paths;
    }

    private static void dfs(TreeNode<Integer> node, List<Integer> currentPath, List<List<Integer>> paths) {
        if (node == null) {
            return;
        }

        currentPath.add(node.val);

        if (node.left == null && node.right == null) {
            paths.add(new ArrayList<>(currentPath));
        } else {
            dfs(node.left, currentPath, paths);
            dfs(node.right, currentPath, paths);
        }

        currentPath.remove(currentPath.size() - 1);
    }

    // S49.
    public static String reverseWords(String input) {
        String[] words = input.split(" ");

        int left = 0;
        int right = words.length - 1;
        while (left < right) {
            String temp = words[left];
            words[left] = words[right];
            words[right] = temp;
            left++;
            right--;
        }

        return String.join(" ", words);
    }

    // S50.
    static class TreeGenerator {
        public static TreeNode<Integer> generate() {
            TreeNode<Integer> root = new TreeNode<>(1);
            root.left = createUnboundedNode();
            root.right = createUnboundedNode();
            return root;
        }

        private static TreeNode<Integer> createUnboundedNode() {
            return new TreeNode<>(-1); // Use a special value to represent an unbounded node
        }
    }

    // S51.
    static class Interval {
        int start;
        int end;

        public Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return "[" + start + ", " + end + "]";
        }
    }

    public static List<Integer> findCoveringSet(List<Interval> intervals) {
        intervals.sort(Comparator.comparingInt(a -> a.start));
        int end = Integer.MIN_VALUE;
        List<Integer> coveringSet = new ArrayList<>();

        for (Interval interval : intervals) {
            if (interval.start > end + 1) {
                coveringSet.add(end);
            }
            end = Math.max(end, interval.end);
        }

        coveringSet.add(end);

        return coveringSet;
    }

    // S52.
    static class TwistedSingleton {
        private static TwistedSingleton instance1;
        private static TwistedSingleton instance2;
        private static int count;

        private TwistedSingleton() {
            // Private constructor to prevent instantiation
        }

        public static synchronized TwistedSingleton getInstance() {
            count++;

            if (count % 2 == 0) {
                if (instance1 == null) {
                    instance1 = new TwistedSingleton();
                }
                return instance1;
            } else {
                if (instance2 == null) {
                    instance2 = new TwistedSingleton();
                }
                return instance2;
            }
        }
    }

    // S53.
    public static int getMaxCoins(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int m = matrix.length;
        int n = matrix[0].length;

        int[][] dp = new int[m][n];

        dp[0][0] = matrix[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + matrix[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j - 1] + matrix[0][j];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = matrix[i][j] + Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }

        return dp[m - 1][n - 1];
    }

    // S54.
    public static void rotateList(List<Integer> list, int k) {
        int n = list.size();
        k = k % n;

        reverse(list, 0, n - 1);
        reverse(list, 0, n - k - 1);
        reverse(list, n - k, n - 1);
    }

    private static void reverse(List<Integer> list, int start, int end) {
        while (start < end) {
            int temp = list.get(start);
            list.set(start, list.get(end));
            list.set(end, temp);
            start++;
            end--;
        }
    }

    // S55.
    public static void solveTowerOfHanoi(int n, int source, int auxiliary, int destination) {
        if (n == 1) {
            System.out.println("Move disk 1 from rod " + source + " to rod " + destination);
            return;
        }

        solveTowerOfHanoi(n - 1, source, destination, auxiliary);
        System.out.println("Move disk " + n + " from rod " + source + " to rod " + destination);
        solveTowerOfHanoi(n - 1, auxiliary, source, destination);
    }

    // S56.
    public static double findSquareRoot(double n) {
        if (n < 0) {
            throw new IllegalArgumentException("Cannot calculate square root of a negative number.");
        }

        if (n == 0) {
            return 0;
        }

        double x = n;
        double y = 0;

        while (x != y) {
            y = x;
            x = (n / x + x) / 2;
        }

        return x;
    }

    // S57.
    public static int getMaxProfit(int[] prices, int k) {
        int n = prices.length;
        int[][] dp = new int[k + 1][n];

        for (int i = 1; i <= k; i++) {
            int maxDiff = -prices[0];

            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], prices[j] + maxDiff);
                maxDiff = Math.max(maxDiff, dp[i - 1][j] - prices[j]);
            }
        }
        return dp[k][n - 1];
    }

    // S58.
    static class SinglyLinkedListNode {
        int val;
        SinglyLinkedListNode next;
        SinglyLinkedListNode random;

        public SinglyLinkedListNode(int val) {
            this.val = val;
        }
    }

    public static SinglyLinkedListNode cloneLinkedList(SinglyLinkedListNode head) {
        if (head == null)
            return null;

        Map<SinglyLinkedListNode, SinglyLinkedListNode> map = new HashMap<>();

        // First pass: create cloned nodes without random pointers
        SinglyLinkedListNode curr = head;
        while (curr != null) {
            SinglyLinkedListNode clone = new SinglyLinkedListNode(curr.val);
            map.put(curr, clone);
            curr = curr.next;
        }

        // Second pass: set random pointers for cloned nodes
        curr = head;
        while (curr != null) {
            SinglyLinkedListNode clone = map.get(curr);
            clone.next = map.get(curr.next);
            clone.random = map.get(curr.random);
            curr = curr.next;
        }

        return map.get(head);
    }

    // S59.
    public static int inorderSuccessor(Node<Integer> root, int target) {
        if (root == null)
            return -1;

        Node<Integer> current = root;
        Node<Integer> successor = null;

        while (current != null) {
            if (current.val > target) {
                successor = current;
                current = current.left;
            } else {
                current = current.right;
            }
        }

        if (successor != null) {
            return successor.val;
        } else {
            return -1;
        }
    }

    // S60.
    public static int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0) {
            return 0;
        }

        int n = heights.length;
        int maxArea = 0;

        // Array to store the indices of the histogram bars
        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i <= n; i++) {
            int height = (i == n) ? 0 : heights[i]; // 1 0 0 0 0 - 2 0 1 1 0 - 3 0 2 2 0 - 0 1 0 0 0

            while (!stack.isEmpty() && height < heights[stack.peek()]) {
                int h = heights[stack.pop()];
                int w = stack.isEmpty() ? i : i - stack.peek() - 1;
                int area = h * w;
                maxArea = Math.max(maxArea, area);
            }
            stack.push(i);
        }

        return maxArea;
    }

    public static int maximalRectangle(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int rows = matrix.length;
        int cols = matrix[0].length;
        int maxArea = 0;
        int[] heights = new int[cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 1) {
                    heights[j] += 1;
                } else {
                    heights[j] = 0;
                }
            }

            int area = largestRectangleArea(heights);
            maxArea = Math.max(maxArea, area);
        }

        return maxArea;
    }

    // S61.
    static class BitArray {
        private int[] arr;

        public BitArray(int size) {
            int length = (size + 31) / 32;
            arr = new int[length];
        }

        public void set(int i, int val) {
            if (val != 0 && val != 1) {
                throw new IllegalArgumentException("Value must be either 0 or 1");
            }

            int index = i / 32; // Calculate the index of the integer
            int bitIndex = i % 32; // Calculate the bit index within the integer

            if (val == 1) {
                arr[index] |= (1 << bitIndex); // Set the bit at the given index
            } else {
                arr[index] &= ~(1 << bitIndex); // Clear the bit at the given index
            }
        }

        public int get(int i) {
            int index = i / 32; // Calculate the index of the integer
            int bitIndex = i % 32; // Calculate the bit index within the integer

            return (arr[index] >> bitIndex) & 1; // Get the value of the bit at the given index
        }
    }

    // S62.
    static class PeekableInterface<T> implements Iterator<T> {
        private Iterator<T> iterator;
        private T nextElement;
        private boolean hasNext;

        public PeekableInterface(Iterator<T> iterator) {
            this.iterator = iterator;
            this.nextElement = null;
            this.hasNext = iterator.hasNext();
            if (hasNext) {
                this.nextElement = iterator.next();
            }
        }

        public T peek() {
            if (!hasNext) {
                throw new NoSuchElementException("No more elements to peek");
            }
            return nextElement;
        }

        @Override
        public T next() {
            if (!hasNext) {
                throw new NoSuchElementException("No more elements");
            }
            T currentElement = nextElement;
            if (iterator.hasNext()) {
                nextElement = iterator.next();
            } else {
                nextElement = null;
                hasNext = false;
            }
            return currentElement;
        }

        @Override
        public boolean hasNext() {
            return hasNext;
        }
    }

    // S63.
    public static int[] findTwoSingleElements(int[] nums) {
        int xor = 0;

        for (int num : nums) {
            xor ^= num;
        }

        // Find the rightmost set bit indicating a difference between two single
        // elements
        int rightmostSetBit = xor & -xor;

        int num1 = 0;
        int num2 = 0;

        // Divide the numbers into two groups based on the rightmost set bit
        for (int num : nums) {
            if ((num & rightmostSetBit) != 0) {
                num1 ^= num;
            } else {
                num2 ^= num;
            }
        }

        return new int[] { num1, num2 };
    }

    // S64.
    public static void partitionList(int[] lst, int x) {
        int low = 0;
        int high = lst.length - 1;
        int i = 0;

        while (i <= high) {
            if (lst[i] < x) {
                swap(lst, i, low);
                i++;
                low++;
            } else if (lst[i] > x) {
                swap(lst, i, high);
                high--;
            } else {
                i++;
            }
        }
    }

    // S65.
    public static Integer findNearestLarger(int[] nums, int index) {
        int left = index - 1;
        int right = index + 1;
        int n = nums.length;
        Integer nearestLargerIndex = null;

        while (left >= 0 || right < n) {
            if (left >= 0 && nums[left] > nums[index]) {
                nearestLargerIndex = left;
                break;
            }
            if (right < n && nums[right] > nums[index]) {
                nearestLargerIndex = right;
                break;
            }
            left--;
            right++;
        }

        return nearestLargerIndex;
    }

    // S66.
    public static TreeNode<Integer> pruneTree(TreeNode<Integer> root) {
        if (root == null) {
            return null;
        }

        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);

        if (root.left == null && root.right == null && root.val == 0) {
            return null;
        }

        return root;
    }

    // S67.
    public static List<String> generateGrayCode(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }

        List<String> grayCode = new ArrayList<>();
        grayCode.add("0");
        grayCode.add("1");

        for (int i = 2; i <= n; i++) {
            int size = grayCode.size();

            for (int j = size - 1; j >= 0; j--) {
                grayCode.add(grayCode.get(j));
            }

            for (int j = 0; j < size; j++) {
                grayCode.set(j, "0" + grayCode.get(j));
                grayCode.set(j + size, "1" + grayCode.get(j + size));
            }
        }

        return grayCode;
    }

    // S68.
    public static void replaceColor(char[][] image, int x, int y, char newColor) {
        int rows = image.length;
        if (rows == 0) {
            return;
        }

        int cols = image[0].length;
        char originalColor = image[x][y];

        if (originalColor == newColor) {
            return;
        }

        replaceColorDFS(image, x, y, originalColor, newColor, rows, cols);
    }

    private static void replaceColorDFS(char[][] image, int x, int y, char originalColor, char newColor, int rows,
            int cols) {
        if (x < 0 || x >= rows || y < 0 || y >= cols || image[x][y] != originalColor) {
            return;
        }

        image[x][y] = newColor;

        replaceColorDFS(image, x - 1, y, originalColor, newColor, rows, cols);
        replaceColorDFS(image, x + 1, y, originalColor, newColor, rows, cols);
        replaceColorDFS(image, x, y - 1, originalColor, newColor, rows, cols);
        replaceColorDFS(image, x, y + 1, originalColor, newColor, rows, cols);
    }

    // S69.
    static class NumberGenerator {
        private int[] numbers;
        private double[] cumulativeProbabilities;
        private Random random;

        public NumberGenerator(int[] numbers, double[] probabilities) {
            if (numbers.length != probabilities.length) {
                throw new IllegalArgumentException("Number of numbers and probabilities must be the same");
            }
            this.numbers = numbers;
            this.cumulativeProbabilities = calculateCumulativeProbabilities(probabilities);
            this.random = new Random();
        }

        public int generateNumberWithProbability() {
            double randomValue = random.nextDouble();
            int index = binarySearch(cumulativeProbabilities, randomValue);
            return numbers[index];
        }

        private double[] calculateCumulativeProbabilities(double[] probabilities) {
            double[] cumulativeProbabilities = new double[probabilities.length];
            cumulativeProbabilities[0] = probabilities[0];
            for (int i = 1; i < probabilities.length; i++) {
                cumulativeProbabilities[i] = cumulativeProbabilities[i - 1] + probabilities[i];
            }
            return cumulativeProbabilities;
        }

        private int binarySearch(double[] arr, double target) {
            int left = 0;
            int right = arr.length - 1;
            while (left < right) {
                int mid = (left + right) / 2;
                if (arr[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            return left;
        }
    }

    // S70.
    public static int findMajorityElement(List<Integer> nums) {
        Map<Integer, Integer> countMap = new HashMap<>();

        for (int num : nums) {
            countMap.put(num, countMap.getOrDefault(num, 0) + 1);
        }

        int majorityElement = 0;
        int majorityCount = 0;

        for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
            int element = entry.getKey();
            int count = entry.getValue();

            if (count > majorityCount) {
                majorityElement = element;
                majorityCount = count;
            }
        }

        return majorityElement;
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
        TreeNode<Integer> binaryTreeRoot = new TreeNode<>(6);
        binaryTreeRoot.left = new TreeNode<>(2);
        binaryTreeRoot.right = new TreeNode<>(8);
        binaryTreeRoot.left.left = new TreeNode<>(1);
        binaryTreeRoot.left.right = new TreeNode<>(4);
        binaryTreeRoot.right.left = new TreeNode<>(7);
        binaryTreeRoot.right.right = new TreeNode<>(9);

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
        int[] deck = new int[52];
        for (int i = 0; i < 52; i++) {
            deck[i] = i + 1;
        }

        shuffleDeck(deck);

        System.out.println("Shuffled deck:");
        for (int card : deck) {
            System.out.print(card + " ");
        }
        System.out.println();

        /*
         * Q22.
         * Implement a queue using two stacks. Recall that a queue is a FIFO (first-in,
         * first-out) data structure with the following methods: enqueue, which inserts
         * an element into the queue, and dequeue, which removes it.
         */
        System.out.println("========= Q22 ==========");
        QueueUsingStacks<Integer> queue = new QueueUsingStacks<>();
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);

        System.out.println(queue.dequeue()); // Output: 1
        System.out.println(queue.dequeue()); // Output: 2

        queue.enqueue(4);
        System.out.println(queue.dequeue()); // Output: 3
        System.out.println(queue.dequeue()); // Output: 4

        System.out.println(queue.isEmpty()); // Output: true

        /*
         * Q23.
         * Given an undirected graph represented as an adjacency matrix and an integer
         * k, write a function to determine whether each vertex in the graph can be
         * colored such that no two adjacent vertices share the same color using at most
         * k colors.
         */
        System.out.println("========= Q23 ==========");
        int[][] graph = {
                { 0, 1, 1, 1 },
                { 1, 0, 1, 0 },
                { 1, 1, 0, 1 },
                { 1, 0, 1, 0 }
        };

        int numOfColours = 3;

        GraphColoring graphColouring = new GraphColoring(graph);
        boolean canColour = graphColouring.canColorGraph(numOfColours);
        System.out.println("Can color graph with " + numOfColours + " colors? " + canColour);

        /*
         * Q24.
         * Given a string s and an integer k, break up the string into multiple lines
         * such that each line has a length of k or less. You must break it up so that
         * words don't break across lines. Each line has to have the maximum possible
         * amount of words. If there's no way to break the text up, then return null.
         * You can assume that there are no spaces at the ends of the string and that
         * there is exactly one space between each word.
         * For example, given the string "the quick brown fox jumps over the lazy dog"
         * and k = 10, you should return: ["the quick", "brown fox", "jumps over",
         * "the lazy", "dog"]. No string in the list has a length of more than 10.
         */
        System.out.println("========= Q24 ==========");
        String s = "the quick brown fox jumps over the lazy dog";
        int lineWidth = 10;

        List<String> lines = breakLines(s, lineWidth);
        if (lines != null) {
            for (String line : lines) {
                System.out.println(line);
            }
        } else {
            System.out.println("Cannot break the text into lines.");
        }

        /*
         * Q25.
         * An sorted array of integers was rotated an unknown number of times.
         * Given such an array, find the index of the element in the array in faster
         * than linear time. If the element doesn't exist in the array, return null.
         * For example, given the array [13, 18, 25, 2, 8, 10] and the element 8, return
         * 4 (the index of 8 in the array).
         * You can assume all the integers in the array are unique.
         */
        System.out.println("========= Q25 ==========");
        int[] nums = { 13, 18, 25, 2, 8, 10 };
        int target = 8;

        Integer index = search(nums, target);
        if (index != null) {
            System.out.println("Index of " + target + " is: " + index);
        } else {
            System.out.println(target + " is not found in the array.");
        }

        /*
         * Q26.
         * Given a multiset of integers, return whether it can be partitioned into two
         * subsets whose sums are the same.
         * For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return
         * true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35}, which
         * both add up to 55.
         * Given the multiset {15, 5, 20, 10, 35}, it would return false, since we can't
         * split it up into two subsets that add up to the same sum.
         */
        System.out.println("========= Q26 ==========");
        int[] numsToPartition = { 15, 5, 20, 10, 35, 15, 10 };
        boolean partitionResult = canPartition(numsToPartition);
        System.out.println("Can partition the multiset: " + partitionResult);

        /*
         * Q27.
         * Implement integer exponentiation. That is, implement the pow(x, y) function,
         * where x and y are integers and returns x^y.
         * Do this faster than the naive method of repeated multiplication.
         * For example, pow(2, 10) should return 1024.
         */
        System.out.println("========= Q27 ==========");
        int x = 2;
        int y = 10;
        long powResult = pow(x, y);
        System.out.println(x + "^" + y + " = " + powResult);

        /*
         * Q28.
         * There is an N by M matrix of zeroes. Given N and M, write a function to count
         * the number of ways of starting at the top-left corner and getting to the
         * bottom-right corner. You can only move right or down.
         * For example, given a 2 by 2 matrix, you should return 2, since there are two
         * ways to get to the bottom-right:
         * Right, then down
         * Down, then right
         * Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.
         */
        System.out.println("========= Q28 ==========");
        int ways = countWays(2, 2);
        System.out.println("Number of ways to reach bottom-right from top-left in 2x2 matrix: " + ways);
        ways = countWays(5, 5);
        System.out.println("Number of ways to reach bottom-right from top-left in 5x5 matrix: " + ways);

        /*
         * Q29.
         * Assume you have access to a function toss_biased() which returns 0 or 1 with
         * a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know
         * the bias of the coin.
         * Write a function to simulate an unbiased coin toss.
         */
        System.out.println("========= Q29 ==========");
        int unbiasedResult = tossUnbiased();
        System.out.println("Unbiased result: " + unbiasedResult);

        /*
         * Q30.
         * On our special chessboard, two bishops attack each other if they share the
         * same diagonal. This includes bishops that have another bishop located between
         * them, i.e. bishops can attack through pieces.
         * You are given N bishops, represented as (row, column) tuples on a M by M
         * chessboard. Write a function to count the number of pairs of bishops that
         * attack each other. The ordering of the pair doesn't matter: (1, 2) is
         * considered the same as (2, 1).
         * For example, given M = 5 and the list of bishops:
         * (0, 0)
         * (1, 2)
         * (2, 2)
         * (4, 0)
         * The board would look like this:
         * [b 0 0 0 0]
         * [0 0 b 0 0]
         * [0 0 b 0 0]
         * [0 0 0 0 0]
         * [b 0 0 0 0]
         * You should return 2, since bishops 1 and 3 attack each other, as well as
         * bishops 3 and 4.
         */
        System.out.println("========= Q30 ==========");
        int M = 5;
        int[][] bishops = {
                { 0, 0 },
                { 1, 2 },
                { 2, 2 },
                { 4, 0 }
        };

        int pairs = countAttackingPairs(bishops, M);
        System.out.println("Number of attacking pairs: " + pairs);

        /*
         * Q31.
         * Suppose you have a multiplication table that is N by N. That is, a 2D array
         * where the value at the i-th row and j-th column is (i + 1) * (j + 1) (if
         * 0-indexed) or i * j (if 1-indexed).
         * Given integers N and X, write a function that returns the number of times X
         * appears as a value in an N by N multiplication table.
         * For example, given N = 6 and X = 12, you should return 4, since the
         * multiplication table looks like this:
         * | 1 | 2 | 3 | 4 | 5 | 6 |
         * | 2 | 4 | 6 | 8 | 10 | 12 |
         * | 3 | 6 | 9 | 12 | 15 | 18 |
         * | 4 | 8 | 12 | 16 | 20 | 24 |
         * | 5 | 10 | 15 | 20 | 25 | 30 |
         * | 6 | 12 | 18 | 24 | 30 | 36 |
         * And there are 4 12's in the table.
         */
        System.out.println("========= Q31 ==========");
        int N = 6;
        int X = 12;

        int occurrences = countOccurrences(N, X);
        System.out.println("Number of occurrences of " + X + ": " + occurrences);

        /*
         * Q32.
         * You are given an N by M 2D matrix of lowercase letters. Determine the minimum
         * number of columns that can be removed to ensure that each row is ordered from
         * top to bottom lexicographically. That is, the letter at each column is
         * lexicographically later as you go down each row. It does not matter whether
         * each row itself is ordered lexicographically.
         * For example, given the following table:
         * cba
         * daf
         * ghi
         * This is not ordered because of the a in the center. We can remove the second
         * column to make it ordered:
         * ca
         * df
         * gi
         * So your function should return 1, since we only needed to remove 1 column.
         * As another example, given the following table:
         * abcdef
         * Your function should return 0, since the rows are already ordered (there's
         * only one row).
         * As another example, given the following table:
         * zyx
         * wvu
         * tsr
         * Your function should return 3, since we would need to remove all the columns
         * to order it.
         */
        System.out.println("========= Q32 ==========");
        String[][] matrix1 = { { "c", "b", "a" }, { "d", "a", "f" }, { "g", "h", "i" } };
        System.out.println("Minimum column removals: " + minColumnRemovals(matrix1));

        String[][] matrix2 = { { "a", "b", "c", "d", "e", "f" } };
        System.out.println("Minimum column removals: " + minColumnRemovals(matrix2));

        String[][] matrix3 = { { "z", "y", "x" }, { "w", "v", "u" }, { "t", "s", "r" } };
        System.out.println("Minimum column removals: " + minColumnRemovals(matrix3));

        /*
         * Q33.
         * Given k sorted singly linked lists, write a function to merge all the lists
         * into one sorted singly linked list.
         */
        System.out.println("========= Q33 ==========");
        ListNode list1 = new ListNode(1);
        list1.next = new ListNode(4);
        list1.next.next = new ListNode(5);

        ListNode list2 = new ListNode(1);
        list2.next = new ListNode(3);
        list2.next.next = new ListNode(4);

        ListNode list3 = new ListNode(2);
        list3.next = new ListNode(6);

        ListNode[] lists = { list1, list2, list3 };

        ListNode mergedList = mergeKLists(lists);
        printList(mergedList);

        /*
         * Q34.
         * Given an array of integers, write a function to determine whether the array
         * could become non-decreasing by modifying at most 1 element.
         * For example, given the array [10, 5, 7], you should return true, since we can
         * modify the 10 into a 1 to make the array non-decreasing.
         * Given the array [10, 5, 1], you should return false, since we can't modify
         * any one element to get a non-decreasing array.
         */
        System.out.println("========= Q34 ==========");
        int[] nums1 = { 10, 5, 7 };
        System.out.println(checkPossibility(nums1)); // Output: true

        int[] nums2 = { 10, 5, 1 };
        System.out.println(checkPossibility(nums2)); // Output: false

        /*
         * Q35.
         * Invert a binary tree.
         * For example, given the following tree:
         * "    a      "
         * "   / \     "
         * "  b   c    "
         * " / \  /    "
         * "d   e f    "
         * should become:
         * "  a        "
         * " / \       "
         * " c  b      "
         * " \  / \    "
         * "  f e  d   "
         */
        System.out.println("========= Q35 ==========");
        TreeNode<Character> a = new TreeNode<>('a');
        TreeNode<Character> b = new TreeNode<>('b');
        TreeNode<Character> c = new TreeNode<>('c');
        TreeNode<Character> d = new TreeNode<>('d');
        TreeNode<Character> e = new TreeNode<>('e');
        TreeNode<Character> f = new TreeNode<>('f');

        a.left = b;
        a.right = c;
        b.left = d;
        b.right = e;
        c.left = f;

        TreeNode<Character> inverted = invertTree(a);

        printTree(inverted);

        /*
         * Q36.
         * Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A
         * 1 represents land and 0 represents water, so an island is a group of 1s that
         * are neighboring whose perimeter is surrounded by water.
         * For example, this matrix has 4 islands.
         * 1 0 0 0 0
         * 0 0 1 1 0
         * 0 1 1 0 0
         * 0 0 0 0 0
         * 1 1 0 0 1
         * 1 1 0 0 1
         */
        System.out.println("========= Q36 ==========");
        int[][] matrix = {
                { 1, 0, 0, 0, 0 },
                { 0, 0, 1, 1, 0 },
                { 0, 1, 1, 0, 0 },
                { 0, 0, 0, 0, 0 },
                { 1, 1, 0, 0, 1 },
                { 1, 1, 0, 0, 1 }
        };

        int islandCount = countIslands(matrix);
        System.out.println("Number of islands: " + islandCount);

        /*
         * Q37.
         * Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0,
         * using only mathematical or bit operations. You can assume b can only be 1 or
         * 0.
         */
        System.out.println("========= Q37 ==========");
        int xForOperation = 10;
        int yForOperation = 20;
        int bForOperation = 1;

        int operationResult = select(xForOperation, yForOperation, bForOperation);
        System.out.println(operationResult); // Output: 10

        /*
         * Q38.
         * Given a string of parentheses, write a function to compute the minimum number
         * of parentheses to be removed to make the string valid (i.e. each open
         * parenthesis is eventually closed).
         * For example, given the string "()())()", you should return 1. Given the
         * string ")(", you should return 2, since we must remove all of them.
         */
        System.out.println("========= Q38 ==========");
        String parentheses1 = "()())()";
        System.out.println(minRemoval(parentheses1)); // Output: 1

        String parentheses2 = ")(";
        System.out.println(minRemoval(parentheses2)); // Output: 2

        /*
         * Q39.
         * Implement division of two positive integers without using the division,
         * multiplication, or modulus operators. Return the quotient as an integer,
         * ignoring the remainder.
         */
        System.out.println("========= Q39 ==========");
        int dividend = 20;
        int divisor = 5;
        System.out.println(divide(dividend, divisor)); // Output: 4

        dividend = 30;
        divisor = 6;
        System.out.println(divide(dividend, divisor)); // Output: 5

        /*
         * Q40.
         * Determine whether a tree is a valid binary search tree.
         * A binary search tree is a tree with two children, left and right, and
         * satisfies the constraint that the key in the left child must be less than or
         * equal to the root and the key in the right child must be greater than or
         * equal to the root.
         */
        System.out.println("========= Q40 ==========");
        BinarySearchTree bst = new BinarySearchTree();

        bst.insert(50);
        bst.insert(30);
        bst.insert(20);
        bst.insert(40);
        bst.insert(70);
        bst.insert(60);
        bst.insert(80);

        System.out.println("Inorder traversal:");
        bst.inorderTraversal();

        int key = 40;
        if (bst.search(key))
            System.out.println("\n" + key + " is present in the BST");
        else
            System.out.println("\n" + key + " is not present in the BST");

        key = 55;
        if (bst.search(key))
            System.out.println(key + " is present in the BST");
        else
            System.out.println(key + " is not present in the BST");

        bst.delete(20);
        System.out.println("\nInorder traversal after deleting 20:");
        bst.inorderTraversal();

        /*
         * Q41.
         * Given an integer n and a list of integers l, write a function that randomly
         * generates a number from 0 to n-1 that isn't in l (uniform).
         */
        System.out.println("========= Q41 ==========");
        int n = 10;
        List<Integer> l = List.of(2, 4, 6, 8);

        int randomNum = generateRandomNumber(n, l);
        System.out.println("Random number: " + randomNum);

        /*
         * Q42.
         * Write a map implementation with a get function that lets you retrieve the
         * value of a key at a particular time.
         * It should contain the following methods:
         * set(key, value, time): sets key to value for t = time.
         * get(key, time): gets the key at t = time.
         * The map should work like this. If we set a key at a particular time, it will
         * maintain that value forever or until it gets set at a later time. In other
         * words, when we get a key at a time, it should return the value that was set
         * for that key set at the most recent time.
         * Consider the following examples:
         * d.set(1, 1, 0) # set key 1 to value 1 at time 0
         * d.set(1, 2, 2) # set key 1 to value 2 at time 2
         * d.get(1, 1) # get key 1 at time 1 should be 1
         * d.get(1, 3) # get key 1 at time 3 should be 2
         * d.set(1, 1, 5) # set key 1 to value 1 at time 5
         * d.get(1, 0) # get key 1 at time 0 should be null
         * d.get(1, 10) # get key 1 at time 10 should be 1
         * d.set(1, 1, 0) # set key 1 to value 1 at time 0
         * d.set(1, 2, 0) # set key 1 to value 2 at time 0
         * d.get(1, 0) # get key 1 at time 0 should be 2
         */
        System.out.println("========= Q42 ==========");
        TimeMap timeMap1 = new TimeMap();
        timeMap1.set(1, 1, 0);
        timeMap1.set(1, 2, 2);
        System.out.println(timeMap1.get(1, 1)); // Output: 1
        System.out.println(timeMap1.get(1, 3)); // Output: 2

        TimeMap timeMap2 = new TimeMap();
        timeMap2.set(1, 1, 5);
        System.out.println(timeMap2.get(1, 0)); // Output: null
        System.out.println(timeMap2.get(1, 10)); // Output: 1

        TimeMap timeMap3 = new TimeMap();
        timeMap3.set(1, 1, 0);
        timeMap3.set(1, 2, 0);
        System.out.println(timeMap3.get(1, 0)); // Output: 2

        /*
         * Q43.
         * Given an unsorted array of integers, find the length of the longest
         * consecutive elements sequence.
         * For example, given [100, 4, 200, 1, 3, 2], the longest consecutive element
         * sequence is [1, 2, 3, 4]. Return its length: 4.
         * Your algorithm should run in O(n) complexity.
         */
        System.out.println("========= Q43 ==========");
        int[] numsToFindLongestConsecutive = { 100, 4, 200, 1, 3, 2 };
        int longestLength = longestConsecutive(numsToFindLongestConsecutive);
        System.out.println("Longest consecutive sequence length: " + longestLength);

        /*
         * Q44.
         * Given a list of integers and a number K, return which contiguous elements of
         * the list sum to K.
         * For example, if the list is [1, 2, 3, 4, 5] and K is 9, then it should return
         * [2, 3, 4], since 2 + 3 + 4 = 9.
         */
        System.out.println("========= Q44 ==========");
        int[] numsToFindContiguousSum = { 1, 2, 3, 4, 5 };
        int targetContiguousSum = 9;
        List<Integer> contiguousElements = findContiguousElementsSum(numsToFindContiguousSum, targetContiguousSum);
        System.out.println(contiguousElements);

        /*
         * Q45.
         * Given a string and a set of characters, return the shortest substring
         * containing all the characters in the set.
         * For example, given the string "figehaeci" and the set of characters {a, e,
         * i}, you should return "aeci".
         * If there is no substring containing all the characters in the set, return
         * null.
         */
        System.out.println("========= Q45 ==========");
        String str = "figehaeci";
        Set<Character> charSet = new HashSet<>(Arrays.asList('a', 'e', 'i'));
        String shortestSubstr = shortestSubstring(str, charSet);
        System.out.println(shortestSubstr);

        /*
         * Q46.
         * Given an integer list where each number represents the number of hops you can
         * make, determine whether you can reach to the last index starting at index 0.
         * For example, [2, 0, 1, 0] returns True while [1, 1, 0, 1] returns False.
         */
        System.out.println("========= Q46 ==========");
        int[] numArr1 = { 2, 0, 1, 0 };
        System.out.println(canReachLastIndex(numArr1)); // true

        int[] numArr2 = { 1, 1, 0, 1 };
        System.out.println(canReachLastIndex(numArr2)); // false

        /*
         * Q47.
         * Given an unsigned 8-bit integer, swap its even and odd bits. The 1st and 2nd
         * bit should be swapped, the 3rd and 4th bit should be swapped, and so on.
         * For example, 10101010 should be 01010101. 11100010 should be 11010001.
         * Bonus: Can you do this in one line?
         */
        System.out.println("========= Q47 ==========");
        int num1 = 0b10101010;
        int binarySwapResult1 = swapEvenOddBits(num1);
        System.out.println(Integer.toBinaryString(binarySwapResult1)); // Output: 01010101

        int num2 = 0b11100010;
        int binarySwapResult2 = swapEvenOddBits(num2);
        System.out.println(Integer.toBinaryString(binarySwapResult2)); // Output: 11010001

        /*
         * Q48.
         * Given a binary tree, return all paths from the root to leaves.
         * For example, given the tree:
         * "   1       "
         * "  / \      "
         * " 2   3     "
         * "    / \    "
         * "   4   5   "
         * Return [[1, 2], [1, 3, 4], [1, 3, 5]].
         */
        System.out.println("========= Q48 ==========");
        TreeNode<Integer> rootToFindAllLeafPath = new TreeNode<>(1);
        rootToFindAllLeafPath.left = new TreeNode<>(2);
        rootToFindAllLeafPath.right = new TreeNode<>(3);
        rootToFindAllLeafPath.right.left = new TreeNode<>(4);
        rootToFindAllLeafPath.right.right = new TreeNode<>(5);

        List<List<Integer>> paths = binaryTreePaths(rootToFindAllLeafPath);

        for (List<Integer> path : paths) {
            System.out.println(path);
        }

        /*
         * Q49.
         * Given a string of words delimited by spaces, reverse the words in string. For
         * example, given "hello world here", return "here world hello"
         * Follow-up: given a mutable string representation, can you perform this
         * operation in-place?
         */
        System.out.println("========= Q49 ==========");
        String input = "hello world here";
        String reversed = reverseWords(input);
        System.out.println(reversed);

        /*
         * Q50.
         * Generate a finite, but an arbitrarily large binary tree quickly in O(1).
         * That is, generate() should return a tree whose size is unbounded but finite.
         */
        System.out.println("========= Q50 ==========");
        // Solution implemented in S50

        /*
         * Q51.
         * Given a set of closed intervals, find the smallest set of numbers that covers
         * all the intervals. If there are multiple smallest sets, return any of them.
         * For example, given the intervals [0, 3], [2, 6], [3, 4], [6, 9], one set of
         * numbers that covers all these intervals is {3, 6}.
         */
        System.out.println("========= Q51 ==========");
        List<Interval> intervals = Arrays.asList(
                new Interval(0, 3),
                new Interval(2, 6),
                new Interval(3, 4),
                new Interval(6, 9));

        List<Integer> coveringSet = findCoveringSet(intervals);
        System.out.println(coveringSet); // Output: [3, 6]

        /*
         * Q52.
         * Implement the singleton pattern with a twist. First, instead of storing one
         * instance, store two instances. And in every even call of getInstance(),
         * return the first instance and in every odd call of getInstance(), return the
         * second instance.
         */
        System.out.println("========= Q52 ==========");
        TwistedSingleton singleton1 = TwistedSingleton.getInstance();
        TwistedSingleton singleton2 = TwistedSingleton.getInstance();
        TwistedSingleton singleton3 = TwistedSingleton.getInstance();

        System.out.println(singleton1); // Output: TwistedSingleton@hashcode
        System.out.println(singleton2); // Output: TwistedSingleton@hashcode
        System.out.println(singleton3); // Output: TwistedSingleton@hashcode

        /*
         * Q53.
         * You are given a 2-d matrix where each cell represents number of coins in that
         * cell. Assuming we start at matrix[0][0], and can only move right or down,
         * find the maximum number of coins you can collect by the bottom right corner.
         * For example, in this matrix
         * 0 3 1 1
         * 2 0 0 4
         * 1 5 3 1
         * The most we can collect is 0 + 2 + 1 + 5 + 3 + 1 = 12 coins.
         */
        System.out.println("========= Q53 ==========");
        int[][] coinMatrix = {
                { 0, 3, 1, 1 },
                { 2, 0, 0, 4 },
                { 1, 5, 3, 1 }
        };

        int maxCoins = getMaxCoins(coinMatrix);
        System.out.println("Maximum number of coins collected: " + maxCoins);

        /*
         * Q54.
         * Write a function that rotates a list by k elements. For example, [1, 2, 3, 4,
         * 5, 6] rotated by two becomes [3, 4, 5, 6, 1, 2]. Try solving this without
         * creating a copy of the list. How many swap or move operations do you need?
         */
        System.out.println("========= Q54 ==========");
        List<Integer> list = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5, 6));
        int rotation = 2;

        System.out.println("Original list: " + list);
        rotateList(list, rotation);
        System.out.println("Rotated list: " + list);

        /*
         * Q55.
         * The Tower of Hanoi is a puzzle game with three rods and n disks, each a
         * different size.
         * All the disks start off on the first rod in a stack. They are ordered by
         * size, with the largest disk on the bottom and the smallest one at the top.
         * The goal of this puzzle is to move all the disks from the first rod to the
         * last rod while following these rules:
         * You can only move one disk at a time.
         * A move consists of taking the uppermost disk from one of the stacks and
         * placing it on top of another stack.
         * You cannot place a larger disk on top of a smaller disk.
         * Write a function that prints out all the steps necessary to complete the
         * Tower of Hanoi. You should assume that the rods are numbered, with the first
         * rod being 1, the second (auxiliary) rod being 2, and the last (goal) rod
         * being 3.
         * For example, with n = 3, we can do this in 7 moves:
         * Move 1 to 3
         * Move 1 to 2
         * Move 3 to 2
         * Move 1 to 3
         * Move 2 to 1
         * Move 2 to 3
         * Move 1 to 3
         */
        System.out.println("========= Q55 ==========");
        int disks = 3;
        int sourceRod = 1;
        int auxiliaryRod = 2;
        int destinationRod = 3;

        solveTowerOfHanoi(disks, sourceRod, auxiliaryRod, destinationRod);

        /*
         * Q56.
         * Given a real number n, find the square root of n. For example, given n = 9,
         * return 3.
         */
        System.out.println("========= Q56 ==========");
        double nForSquareRoot = 9;
        double squareRoot = findSquareRoot(nForSquareRoot);
        System.out.println("The square root of " + n + " is: " + squareRoot);

        /*
         * Q57.
         * Given an array of numbers representing the stock prices of a company in
         * chronological order and an integer k, return the maximum profit you can make
         * from k buys and sells. You must buy the stock before you can sell it, and you
         * must sell the stock before you can buy it again.
         * For example, given k = 2 and the array [5, 2, 4, 0, 1], you should return 3.
         */
        System.out.println("========= Q57 ==========");
        int[] prices = { 5, 2, 4, 0, 1 };
        int numOfBuy = 2;
        int maxProfit = getMaxProfit(prices, numOfBuy);
        System.out.println("Maximum profit: " + maxProfit);

        /*
         * Q58.
         * Given the head to a singly linked list, where each node also has a “random”
         * pointer that points to anywhere in the linked list, deep clone the list.
         */
        System.out.println("========= Q58 ==========");
        SinglyLinkedListNode singlyListHead = new SinglyLinkedListNode(1);
        singlyListHead.next = new SinglyLinkedListNode(2);
        singlyListHead.next.next = new SinglyLinkedListNode(3);
        singlyListHead.next.next.next = new SinglyLinkedListNode(4);
        singlyListHead.next.next.next.next = new SinglyLinkedListNode(5);

        singlyListHead.random = singlyListHead.next.next;
        singlyListHead.next.random = singlyListHead;
        singlyListHead.next.next.random = singlyListHead.next.next.next.next;
        singlyListHead.next.next.next.random = singlyListHead.next.next;
        singlyListHead.next.next.next.next.random = singlyListHead.next;

        SinglyLinkedListNode clonedHead = cloneLinkedList(singlyListHead);

        SinglyLinkedListNode curr = clonedHead;
        while (curr != null) {
            System.out.println("Node value: " + curr.val + " ");
            System.out.println("Node random value: " + curr.random.val + " ");
            curr = curr.next;
        }

        /*
         * Q59.
         * Given a node in a binary search tree, return the next bigger element, also
         * known as the inorder successor.
         * For example, the inorder successor of 22 is 30.
         * "   10          "
         * "  /  \         "
         * " 5    30       "
         * "     /  \      "
         * "   22    35    "
         * You can assume each node has a parent pointer.
         */
        System.out.println("========= Q59 ==========");
        Node<Integer> bstRootToFindInorderSuccessor = new Node<>(10);
        bstRootToFindInorderSuccessor.left = new Node<>(5);
        bstRootToFindInorderSuccessor.right = new Node<>(30);
        bstRootToFindInorderSuccessor.right.left = new Node<>(22);
        bstRootToFindInorderSuccessor.right.right = new Node<>(35);

        int targetForInorderSuccessor = 22;
        int successor = inorderSuccessor(bstRootToFindInorderSuccessor, targetForInorderSuccessor);

        if (successor != -1) {
            System.out.println("The inorder successor of " + targetForInorderSuccessor + " is " + successor);
        } else {
            System.out.println("No inorder successor found for " + targetForInorderSuccessor);
        }

        /*
         * Q60.
         * Given an N by M matrix consisting only of 1's and 0's, find the largest
         * rectangle containing only 1's and return its area.
         * For example, given the following matrix:
         * [[1, 0, 0, 0],
         * [1, 0, 1, 1],
         * [1, 0, 1, 1],
         * [0, 1, 0, 0]]
         * Return 4.
         */
        System.out.println("========= Q60 ==========");
        int[][] rectangleMatrix = {
                { 1, 0, 0, 0 },
                { 1, 0, 1, 1 },
                { 1, 0, 1, 1 },
                { 0, 1, 0, 0 }
        };

        int maxArea = maximalRectangle(rectangleMatrix);
        System.out.println("The largest rectangle area containing only 1's is: " + maxArea);

        /*
         * Q61.
         * Implement a bit array.
         * A bit array is a space efficient array that holds a value of 1 or 0 at each
         * index.
         * init(size): initialize the array with size
         * set(i, val): updates index at i with val where val is either 1 or 0.
         * get(i): gets the value at index i.
         */
        System.out.println("========= Q61 ==========");
        BitArray bitArray = new BitArray(10);
        bitArray.set(3, 1);
        bitArray.set(7, 1);
        bitArray.set(9, 1);

        System.out.println(bitArray.get(3)); // Output: 1
        System.out.println(bitArray.get(7)); // Output: 1
        System.out.println(bitArray.get(9)); // Output: 1
        System.out.println(bitArray.get(0)); // Output: 0
        System.out.println(bitArray.get(5)); // Output: 0

        /*
         * Q62.
         * Given an iterator with methods next() and hasNext(), create a wrapper
         * iterator, PeekableInterface, which also implements peek(). peek shows the
         * next element that would be returned on next().
         * Here is the interface:
         * "class PeekableInterface(object):   "
         * "    def __init__(self, iterator):  "
         * "        pass                       "
         * "    def peek(self):                "
         * "        pass                       "
         * "    def next(self):                "
         * "        pass                       "
         * "    def hasNext(self):             "
         * "        pass                       "
         */
        System.out.println("========= Q62 ==========");
        Integer[] arr = { 1, 2, 3, 4, 5 };
        Iterator<Integer> iterator = Arrays.asList(arr).iterator();

        PeekableInterface<Integer> peekable = new PeekableInterface<>(iterator);
        System.out.println(peekable.peek()); // Output: 1
        System.out.println(peekable.next()); // Output: 1
        System.out.println(peekable.next()); // Output: 2
        System.out.println(peekable.peek()); // Output: 3
        System.out.println(peekable.next()); // Output: 3
        System.out.println(peekable.hasNext()); // Output: true

        /*
         * Q63.
         * Given an array of integers in which two elements appear exactly once and all
         * other elements appear exactly twice, find the two elements that appear only
         * once.
         * For example, given the array [2, 4, 6, 8, 10, 2, 6, 10], return 4 and 8. The
         * order does not matter.
         * Follow-up: Can you do this in linear time and constant space?
         */
        System.out.println("========= Q63 ==========");
        int[] numArr = { 2, 4, 6, 8, 10, 2, 6, 10 };
        int[] singleAppearances = findTwoSingleElements(numArr);
        System.out.println("Single elements: " + Arrays.toString(singleAppearances));

        /*
         * Q64.
         * Given a pivot x, and a list lst, partition the list into three parts.
         * The first part contains all elements in lst that are less than x
         * The second part contains all elements in lst that are equal to x
         * The third part contains all elements in lst that are larger than x
         * Ordering within a part can be arbitrary.
         * For example, given x = 10 and lst = [9, 12, 3, 5, 14, 10, 10], one partition
         * may be [9, 3, 5, 10, 10, 12, 14].
         */
        System.out.println("========= Q64 ==========");
        int pivot = 10;
        int[] lst = { 9, 12, 3, 5, 14, 10, 10 };
        partitionList(lst, pivot);
        System.out.println("Partitioned List: " + Arrays.toString(lst));

        /*
         * Q65.
         * Given an array of numbers and an index i, return the index of the nearest
         * larger number of the number at index i, where distance is measured in array
         * indices.
         * For example, given [4, 1, 3, 5, 6] and index 0, you should return 3.
         * If two distances to larger numbers are the equal, then return any one of
         * them. If the array at i doesn't have a nearest larger integer, then return
         * null.
         * Follow-up: If you can preprocess the array, can you do this in constant time?
         */
        System.out.println("========= Q65 ==========");
        int[] numsToFindNearestLarger = { 4, 1, 3, 5, 6 };
        int idx = 0;

        Integer nearestLargerIndex = findNearestLarger(numsToFindNearestLarger, idx);
        if (nearestLargerIndex != null) {
            System.out.println("Nearest larger number index: " + nearestLargerIndex);
        } else {
            System.out.println("No nearest larger number found.");
        }

        /*
         * Q66.
         * Given a binary tree where all nodes are either 0 or 1, prune the tree so that
         * subtrees containing all 0s are removed.
         * For example, given the following tree:
         * "   0       "
         * "  / \      "
         * " 1   0     "
         * "    / \    "
         * "   1   0   "
         * "  / \      "
         * " 0   0     "
         * should be pruned to:
         * "   0       "
         * "  / \      "
         * " 1   0     "
         * "    /      "
         * "   1       "
         * We do not remove the tree at the root or its left child because it still has
         * a 1 as a descendant.
         */
        System.out.println("========= Q66 ==========");
        TreeNode<Integer> rootToPrune = new TreeNode<>(0);
        rootToPrune.left = new TreeNode<>(1);
        rootToPrune.right = new TreeNode<>(0);
        rootToPrune.right.left = new TreeNode<>(1);
        rootToPrune.right.right = new TreeNode<>(0);
        rootToPrune.right.left.left = new TreeNode<>(0);
        rootToPrune.right.left.right = new TreeNode<>(0);

        TreeNode<Integer> prunedTree = pruneTree(rootToPrune);

        printTree(prunedTree);

        /*
         * Q67.
         * https://en.wikipedia.org/wiki/Gray_code
         * Gray code is a binary code where each successive value differ in only one
         * bit, as well as when wrapping around. Gray code is common in hardware so that
         * we don't see temporary spurious values during transitions.
         * Given a number of bits n, generate a possible gray code for it.
         * For example, for n = 2, one gray code would be [00, 01, 11, 10].
         */
        System.out.println("========= Q67 ==========");
        int nForGrayCode = 2;
        List<String> grayCode = generateGrayCode(nForGrayCode);

        for (String code : grayCode) {
            System.out.println(code);
        }

        /*
         * Q68.
         * Given a 2-D matrix representing an image, a location of a pixel in the screen
         * and a color C, replace the color of the given pixel and all adjacent same
         * colored pixels with C.
         * For example, given the following matrix, and location pixel of (2, 2), and
         * 'G' for green:
         * B B W
         * W W W
         * W W W
         * B B B
         * Becomes
         * B B G
         * G G G
         * G G G
         * B B B
         */
        System.out.println("========= Q68 ==========");
        char[][] image = {
                { 'B', 'B', 'W' },
                { 'W', 'W', 'W' },
                { 'W', 'W', 'W' },
                { 'B', 'B', 'B' }
        };

        int[] pxLocation = { 2, 2 };
        char newColor = 'G';

        replaceColor(image, pxLocation[0], pxLocation[1], newColor);

        for (char[] row : image) {
            for (char pixel : row) {
                System.out.print(pixel + " ");
            }
            System.out.println();
        }

        /*
         * Q69.
         * You are given n numbers as well as n probabilities that sum up to 1. Write a
         * function to generate one of the numbers with its corresponding probability.
         * For example, given the numbers [1, 2, 3, 4] and probabilities [0.1, 0.5, 0.2,
         * 0.2], your function should return 1 10% of the time, 2 50% of the time, and 3
         * and 4 20% of the time.
         * You can generate random numbers between 0 and 1 uniformly.
         */
        System.out.println("========= Q69 ==========");
        int[] numbers = { 1, 2, 3, 4 };
        double[] probabilities = { 0.1, 0.5, 0.2, 0.2 };

        NumberGenerator generator = new NumberGenerator(numbers, probabilities);

        for (int i = 0; i < 10; i++) {
            int number = generator.generateNumberWithProbability();
            System.out.println("Generated number: " + number);
        }

        /*
         * Q70.
         * Given a list of elements, find the majority element, which appears more than
         * half the time (> floor(len(lst) / 2.0)).
         * You can assume that such element exists.
         * For example, given [1, 2, 1, 1, 3, 4, 0], return 1.
         */
        System.out.println("========= Q70 ==========");
        List<Integer> numsToFindMajority = List.of(1, 2, 1, 1, 3, 4, 0);
        int majorityElement = findMajorityElement(numsToFindMajority);
        System.out.println("Majority Element: " + majorityElement);

    }
}
