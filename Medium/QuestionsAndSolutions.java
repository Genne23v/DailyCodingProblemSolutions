package Medium;

import java.util.Random;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.EmptyStackException;
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

    public static <T> void printInorder(TreeNode<T> root) {
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

    // S71.
    public static int findSmallestSquaredSum(int n) {
        int[] dp = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            dp[i] = Integer.MAX_VALUE;

            for (int j = 1; j * j <= i; j++) {
                // Update dp[i] by considering the minimum of dp[i - j*j] + 1
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }

        return dp[n];
    }

    // S72.
    public static int countPaths(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        int[][] dp = new int[m][n];

        // Initialize the first row and first column
        dp[0][0] = 1;
        for (int i = 1; i < m; i++) {
            if (matrix[i][0] == 1) {
                break;
            }
            dp[i][0] = 1;
        }
        for (int j = 1; j < n; j++) {
            if (matrix[0][j] == 1) {
                break;
            }
            dp[0][j] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 1) {
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }

    // S73.
    static class WordTrieNode {
        Map<Character, WordTrieNode> children;
        int count;

        public WordTrieNode() {
            this.children = new HashMap<>();
            this.count = 0;
        }
    }

    public static List<String> findShortestUniquePrefix(String[] words) {
        WordTrieNode root = new WordTrieNode();
        buildTrie(root, words);

        List<String> result = new ArrayList<>();
        for (String word : words) {
            String prefix = findPrefix(root, word);
            result.add(prefix);
        }
        return result;
    }

    private static void buildTrie(WordTrieNode root, String[] words) {
        for (String word : words) {
            WordTrieNode current = root;
            for (char c : word.toCharArray()) {
                current.count++;
                current.children.putIfAbsent(c, new WordTrieNode());
                current = current.children.get(c);
            }
            current.count++;
        }
    }

    private static String findPrefix(WordTrieNode root, String word) {
        StringBuilder prefix = new StringBuilder();
        WordTrieNode current = root;
        for (char c : word.toCharArray()) {
            prefix.append(c);
            current = current.children.get(c);
            if (current.count == 1) {
                break;
            }
        }
        return prefix.toString();
    }

    // S74.
    public static int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];

        // Move slow pointer by one step and fast pointer by two steps
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        return slow;
    }

    // S75.
    static class Element {
        int value;
        int index;

        public Element(int value, int index) {
            this.value = value;
            this.index = index;
        }
    }

    private static void mergeSort(Element[] elements, int start, int end, int[] counts) {
        if (start >= end) {
            return;
        }

        int mid = start + (end - start) / 2;

        mergeSort(elements, start, mid, counts);
        mergeSort(elements, mid + 1, end, counts);

        merge(elements, start, mid, end, counts);
    }

    private static void merge(Element[] elements, int start, int mid, int end, int[] counts) {
        int leftSize = mid - start + 1;
        Element[] leftElements = new Element[leftSize];
        System.arraycopy(elements, start, leftElements, 0, leftSize);

        int rightSize = end - mid;
        Element[] rightElements = new Element[rightSize];
        System.arraycopy(elements, mid + 1, rightElements, 0, rightSize);

        int i = 0, j = 0, k = start, smallerCount = 0;

        while (i < leftSize && j < rightSize) {
            if (leftElements[i].value <= rightElements[j].value) {
                elements[k] = leftElements[i];
                counts[leftElements[i].index] += smallerCount;
                i++;
            } else {
                elements[k] = rightElements[j];
                smallerCount++;
                j++;
            }
            k++;
        }

        while (i < leftSize) {
            elements[k] = leftElements[i];
            counts[leftElements[i].index] += smallerCount;
            i++;
            k++;
        }

        while (j < rightSize) {
            elements[k] = rightElements[j];
            j++;
            k++;
        }
    }

    public static int[] countSmallerElements(int[] nums) {
        int[] counts = new int[nums.length];
        Element[] elements = new Element[nums.length];

        for (int i = 0; i < nums.length; i++) {
            elements[i] = new Element(nums[i], i);
        }

        mergeSort(elements, 0, nums.length - 1, counts);

        return counts;
    }

    // S76.
    static class TwoDIterator<T> {
        private Iterator<List<T>> rowIterator;
        private Iterator<T> colIterator;

        public TwoDIterator(List<List<T>> arrays) {
            rowIterator = arrays.iterator();
            colIterator = null;
        }

        public boolean hasNext() {
            if (colIterator == null || !colIterator.hasNext()) {
                while (rowIterator.hasNext()) {
                    List<T> row = rowIterator.next();
                    if (!row.isEmpty()) {
                        colIterator = row.iterator();
                        return true;
                    }
                }
                return false;
            }
            return true;
        }

        public T next() {
            if (hasNext()) {
                return colIterator.next();
            }
            throw new RuntimeException("No more elements");
        }
    }

    // S77.
    public static void rotateMatrix(int[][] matrix) {
        int n = matrix.length;

        // Transpose the matrix
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        // Reverse each row
        for (int i = 0; i < n; i++) {
            int start = 0;
            int end = n - 1;
            while (start < end) {
                int temp = matrix[i][start];
                matrix[i][start] = matrix[i][end];
                matrix[i][end] = temp;
                start++;
                end--;
            }
        }
    }

    // S78.
    public static ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode middle = getMiddle(head);
        ListNode nextOfMiddle = middle.next;

        // Split the list into two halves
        middle.next = null;

        ListNode left = sortList(head);
        ListNode right = sortList(nextOfMiddle);

        ListNode sortedList = merge(left, right);

        return sortedList;
    }

    private static ListNode getMiddle(ListNode head) {
        if (head == null) {
            return null;
        }

        ListNode slow = head;
        ListNode fast = head.next;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        return slow;
    }

    private static ListNode merge(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;

        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                curr.next = l1;
                l1 = l1.next;
            } else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }

        if (l1 != null) {
            curr.next = l1;
        } else if (l2 != null) {
            curr.next = l2;
        }

        return dummy.next;
    }

    // S79.
    static class WordTransformation {
        public List<String> findTransformation(String start, String end, Set<String> dictionary) {
            dictionary.add(start);

            // Build the adjacency graph
            Map<String, List<String>> graph = buildGraph(dictionary);

            // Perform BFS traversal
            Map<String, String> parentMap = new HashMap<>();
            Queue<String> queue = new LinkedList<>();
            Set<String> visited = new HashSet<>();

            queue.offer(start);
            visited.add(start);

            while (!queue.isEmpty()) {
                String current = queue.poll();

                if (current.equals(end)) {
                    return constructPath(parentMap, start, end);
                }

                List<String> transformations = graph.getOrDefault(current, new ArrayList<>());

                for (String word : transformations) {
                    if (!visited.contains(word)) {
                        parentMap.put(word, current);
                        visited.add(word);
                        queue.offer(word);
                    }
                }
            }
            return null;
        }

        private Map<String, List<String>> buildGraph(Set<String> dictionary) {
            Map<String, List<String>> graph = new HashMap<>();

            for (String word : dictionary) {
                graph.put(word, new ArrayList<>());

                char[] chars = word.toCharArray();
                for (int i = 0; i < chars.length; i++) {
                    char originalChar = chars[i];

                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c == originalChar) {
                            continue;
                        }

                        chars[i] = c;
                        String transformedWord = String.valueOf(chars);

                        if (dictionary.contains(transformedWord)) {
                            graph.get(word).add(transformedWord);
                        }
                    }

                    chars[i] = originalChar;
                }
            }

            return graph;
        }

        private List<String> constructPath(Map<String, String> parentMap, String start, String end) {
            List<String> path = new ArrayList<>();
            String current = end;

            while (current != null) {
                path.add(0, current);
                current = parentMap.get(current);
            }

            return path;
        }
    }

    // S80.
    public static List<Integer> findSubstring(String s, String[] words) {
        List<Integer> result = new ArrayList<>();
        if (s == null || s.length() == 0 || words == null || words.length == 0) {
            return result;
        }

        int wordLength = words[0].length();
        int totalLength = wordLength * words.length;

        Map<String, Integer> wordCount = new HashMap<>();
        for (String word : words) {
            wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
        }

        for (int i = 0; i <= s.length() - totalLength; i++) {
            Map<String, Integer> currentCount = new HashMap<>();
            int j = 0;

            while (j < words.length) {
                String word = s.substring(i + j * wordLength, i + (j + 1) * wordLength);
                if (!wordCount.containsKey(word)) {
                    break;
                }

                currentCount.put(word, currentCount.getOrDefault(word, 0) + 1);

                if (currentCount.get(word) > wordCount.get(word)) {
                    break;
                }

                j++;
            }

            if (j == words.length) {
                result.add(i);
            }
        }

        return result;
    }

    // S81.
    // Ad-hoc polymorphism through function overloading (or operator overloading)
    class Calculator {
        public int add(int a, int b) {
            return a + b;
        }

        public double add(double a, double b) {
            return a + b;
        }

        public String add(String a, String b) {
            return a.concat(b);
        }
    }

    // Parametric polymorphism through generics
    class GenericStack<T> {
        private List<T> elements = new ArrayList<>();

        public void push(T element) {
            elements.add(element);
        }

        public T pop() {
            if (elements.isEmpty()) {
                throw new EmptyStackException();
            }
            return elements.remove(elements.size() - 1);
        }
    }

    // Subtype polymorphism through inheritance and method overriding
    class Animal {
        public void makeSound() {
            System.out.println("Animal is making a sound");
        }
    }

    class Dog extends Animal {
        @Override
        public void makeSound() {
            System.out.println("Dog is barking");
        }
    }

    class Cat extends Animal {
        @Override
        public void makeSound() {
            System.out.println("Cat is meowing");
        }
    }

    // S82.
    public static TreeNode<Integer> buildTree(int[] postorder, int start, int end) {
        if (start > end) {
            return null;
        }

        int rootVal = postorder[end];
        TreeNode<Integer> root = new TreeNode<>(rootVal);

        // Find the index of the last element smaller than the root value
        int i;
        for (i = end - 1; i >= start; i--) {
            if (postorder[i] < rootVal) {
                break;
            }
        }

        root.left = buildTree(postorder, start, i);
        root.right = buildTree(postorder, i + 1, end - 1);

        return root;
    }

    public static TreeNode<Integer> buildTreeFromPostorder(int[] postorder) {
        if (postorder == null || postorder.length == 0) {
            return null;
        }

        return buildTree(postorder, 0, postorder.length - 1);
    }

    public static void inorderTraversal(TreeNode<Integer> node) {
        if (node == null) {
            return;
        }

        inorderTraversal(node.left);
        System.out.print(node.val + " ");
        inorderTraversal(node.right);
    }

    // S83.
    public static void interleaveStack(Stack<Integer> stack) {
        int size = stack.size();
        int halfSize = size / 2;

        Queue<Integer> queue = new LinkedList<>();
        Stack<Integer> tempStack = new Stack<>();

        for (int i = 0; i < halfSize; i++) {
            queue.add(stack.pop());
        }

        while (!stack.isEmpty()) {
            tempStack.push(stack.pop());
        }

        while (!queue.isEmpty()) {
            stack.push(tempStack.pop());
            stack.push(queue.poll());
        }

        while (!tempStack.isEmpty()) {
            stack.push(tempStack.pop());
        }
    }

    // S84.
    static class Graph {
        private int numVertices;
        private List<List<Integer>> adjList;

        // Member variables for S113.
        private int[] disc; // Discovery time of each vertex in DFS traversal
        private int[] low; // Earliest reachable vertex
        private int[] parent;
        private int time;
        private boolean[] visited;
        private List<int[]> bridges;

        public Graph(int numVertices) {
            this.numVertices = numVertices;
            adjList = new ArrayList<>(numVertices);
            for (int i = 0; i < numVertices; i++) {
                adjList.add(new ArrayList<>());
            }

            // Member variables initialization for S113
            disc = new int[numVertices];
            low = new int[numVertices];
            parent = new int[numVertices];
            visited = new boolean[numVertices];
            bridges = new ArrayList<>();
        }

        public void addEdge(int u, int v) {
            adjList.get(u).add(v);
            adjList.get(v).add(u);
        }

        public boolean isMinimallyConnected() {
            boolean[] visited = new boolean[numVertices];

            return dfs(0, visited, -1) && allVisited(visited);
        }

        private boolean dfs(int vertex, boolean[] visited, int parent) {
            visited[vertex] = true;

            for (int neighbor : adjList.get(vertex)) {
                if (!visited[neighbor]) {
                    if (!dfs(neighbor, visited, vertex)) {
                        return false;
                    }
                } else if (neighbor != parent) {
                    return false;
                }
            }
            return true;
        }

        private boolean allVisited(boolean[] visited) {
            for (boolean v : visited) {
                if (!v) {
                    return false;
                }
            }
            return true;
        }

        // S90.
        public boolean isBipartite() {
            int[] colors = new int[numVertices];
            Arrays.fill(colors, -1);

            for (int i = 0; i < numVertices; i++) {
                if (colors[i] == -1) {
                    if (!isBipartiteUtil(i, colors)) {
                        return false;
                    }
                }
            }

            return true;
        }

        private boolean isBipartiteUtil(int src, int[] colors) {
            Queue<Integer> queue = new LinkedList<>();
            queue.offer(src);
            colors[src] = 1;

            while (!queue.isEmpty()) {
                int curr = queue.poll();

                for (int neighbor : adjList.get(curr)) {
                    if (colors[neighbor] == -1) {
                        colors[neighbor] = 1 - colors[curr];
                        queue.offer(neighbor);
                    } else if (colors[neighbor] == colors[curr]) {
                        return false;
                    }
                }
            }

            return true;
        }

        // S113.
        public List<int[]> findBridges() {
            for (int i = 0; i < numVertices; i++) {
                if (!visited[i]) {
                    dfs(i);
                }
            }
            return bridges;
        }

        private void dfs(int u) {
            visited[u] = true;
            disc[u] = low[u] = ++time;

            for (int v : adjList.get(u)) {
                if (!visited[v]) {
                    parent[v] = u;
                    dfs(v);
                    low[u] = Math.min(low[u], low[v]);

                    if (low[v] > disc[u]) {
                        // No other path can reach v except through u
                        bridges.add(new int[] { u, v });
                    }
                } else if (v != parent[u]) {
                    low[u] = Math.min(low[u], disc[v]);
                }
            }
        }
    }

    // S85.
    // Solution in Q85

    // S86.
    public static int getMaxSubarraySum(int[] nums) {
        int maxSum = nums[0];
        int currentMax = nums[0];
        int minSum = nums[0];
        int currentMin = nums[0];
        int totalSum = nums[0];

        for (int i = 1; i < nums.length; i++) {
            totalSum += nums[i];

            currentMax = Math.max(nums[i], currentMax + nums[i]);
            maxSum = Math.max(maxSum, currentMax);

            currentMin = Math.min(nums[i], currentMin + nums[i]);
            minSum = Math.min(minSum, currentMin);
        }

        // If the total sum equals the minimum subarray sum,
        // it means all elements in the array are negative, so return the maximum
        // subarray sum
        if (totalSum == minSum) {
            return maxSum;
        }

        // Otherwise, return the maximum of the maximum subarray sum and the difference
        // between the total sum and the minimum subarray sum
        return Math.max(maxSum, totalSum - minSum);
    }

    // S87.
    public static boolean canReachEnd(int[] nums) {
        int lastReachableIndex = nums.length - 1;

        for (int i = nums.length - 2; i >= 0; i--) {
            if (i + nums[i] >= lastReachableIndex) {
                lastReachableIndex = i;
            }
        }

        return lastReachableIndex == 0;
    }

    // S88.
    public static List<Integer> largestDivisibleSubset(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }

        Arrays.sort(nums);

        int n = nums.length;
        int[] dp = new int[n]; // Stores the size of the largest subset ending at index i
        int[] prev = new int[n]; // Stores the index of the previous element in the subset

        int maxSize = 0;
        int maxIdx = 0;

        for (int i = 0; i < n; i++) {
            dp[i] = 1;
            prev[i] = -1;

            for (int j = 0; j < i; j++) {
                if (nums[i] % nums[j] == 0 && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    prev[i] = j;
                }
            }

            if (dp[i] > maxSize) {
                maxSize = dp[i];
                maxIdx = i;
            }
        }

        List<Integer> result = new ArrayList<>();
        while (maxIdx != -1) {
            result.add(nums[maxIdx]);
            maxIdx = prev[maxIdx];
        }

        return result;
    }

    // S89.
    public static int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            // Check if the mid element is greater than the rightmost element
            // If true, the minimum element is on the right side of mid
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            }
            // If false, the minimum element is on the left side of mid
            else {
                right = mid;
            }
        }

        // left and right will converge to the minimum element
        return nums[left];
    }

    // S90.
    // Solution added to Graph class in S84

    // S91.
    public static ListNode partition(ListNode head, int k) {
        ListNode smallerHead = new ListNode(0);
        ListNode smallerTail = smallerHead;
        ListNode greaterHead = new ListNode(0);
        ListNode greaterTail = greaterHead;

        ListNode curr = head;

        while (curr != null) {
            if (curr.val < k) {
                smallerTail.next = curr;
                smallerTail = smallerTail.next;
            } else {
                greaterTail.next = curr;
                greaterTail = greaterTail.next;
            }

            curr = curr.next;
        }

        greaterTail.next = null;
        smallerTail.next = greaterHead.next;

        return smallerHead.next;
    }

    // S92.
    public static List<Integer> findPatternIndices(String str, String pattern) {
        List<Integer> indices = new ArrayList<>();

        int n = str.length();
        int m = pattern.length();

        for (int i = 0; i <= n - m; i++) {
            int j;
            for (j = 0; j < m; j++) {
                if (str.charAt(i + j) != pattern.charAt(j))
                    break;
            }

            if (j == m) {
                indices.add(i);
            }
        }

        return indices;
    }

    // S93.
    public static List<String> generateIPAddresses(String s) {
        List<String> result = new ArrayList<>();
        backtrack(s, 0, new ArrayList<>(), result);
        return result;
    }

    private static void backtrack(String s, int index, List<String> current, List<String> result) {
        if (index == s.length() && current.size() == 4) {
            result.add(String.join(".", current));
        } else if (index < s.length() && current.size() < 4) {
            // Try different substrings of s starting from the current index
            for (int i = 1; i <= 3 && index + i <= s.length(); i++) {
                String segment = s.substring(index, index + i);
                if (isValidSegment(segment)) {
                    current.add(segment);
                    backtrack(s, index + i, current, result);
                    current.remove(current.size() - 1);
                }
            }
        }
    }

    private static boolean isValidSegment(String segment) {
        if (segment.length() > 1 && segment.charAt(0) == '0') {
            return false;
        }

        int value = Integer.parseInt(segment);
        return value >= 0 && value <= 255;
    }

    // S94.
    static class NodeWithHorizontalDistance {
        int value;
        int hd;
        NodeWithHorizontalDistance left, right;

        public NodeWithHorizontalDistance(int value) {
            this.value = value;
            this.hd = Integer.MAX_VALUE;
            this.left = null;
            this.right = null;
        }
    }

    public static List<Integer> bottomView(NodeWithHorizontalDistance root) {
        List<Integer> result = new ArrayList<>();
        if (root == null)
            return result;

        Map<Integer, Integer> map = new TreeMap<>();

        Queue<NodeWithHorizontalDistance> queue = new LinkedList<>();
        root.hd = 0;
        queue.add(root);

        while (!queue.isEmpty()) {
            NodeWithHorizontalDistance node = queue.poll();
            map.put(node.hd, node.value);

            if (node.left != null) {
                node.left.hd = node.hd - 1;
                queue.add(node.left);
            }
            if (node.right != null) {
                node.right.hd = node.hd + 1;
                queue.add(node.right);
            }
        }

        for (int value : map.values()) {
            result.add(value);
        }

        return result;
    }

    // S95.
    public static int romanToDecimal(String roman) {
        Map<Character, Integer> symbolValues = createSymbolValuesMap();
        int decimal = 0;

        for (int i = 0; i < roman.length(); i++) {
            int currentValue = symbolValues.get(roman.charAt(i));

            if (i + 1 < roman.length() && symbolValues.get(roman.charAt(i + 1)) > currentValue) {
                decimal -= currentValue;
            } else {
                decimal += currentValue;
            }
        }

        return decimal;
    }

    private static Map<Character, Integer> createSymbolValuesMap() {
        Map<Character, Integer> symbolValues = new HashMap<>();
        symbolValues.put('M', 1000);
        symbolValues.put('D', 500);
        symbolValues.put('C', 100);
        symbolValues.put('L', 50);
        symbolValues.put('X', 10);
        symbolValues.put('V', 5);
        symbolValues.put('I', 1);
        return symbolValues;
    }

    // S96.
    static class ReversingGraph {
        private Map<String, List<String>> adjacencyList;

        public ReversingGraph() {
            adjacencyList = new HashMap<>();
        }

        public void addEdge(String source, String destination) {
            adjacencyList.computeIfAbsent(source, k -> new ArrayList<>()).add(destination);
        }

        public ReversingGraph reverse() {
            ReversingGraph reversedGraph = new ReversingGraph();

            for (String vertex : adjacencyList.keySet()) {
                for (String destination : adjacencyList.get(vertex)) {
                    reversedGraph.addEdge(destination, vertex);
                }
            }

            return reversedGraph;
        }

        public void printGraph() {
            for (String vertex : adjacencyList.keySet()) {
                System.out.print(vertex + " -> ");
                List<String> neighbors = adjacencyList.get(vertex);
                for (String neighbor : neighbors) {
                    System.out.print(neighbor + " ");
                }
                System.out.println();
            }
        }
    }

    // S97.
    public static int maxMoney(int[] coins) {
        int n = coins.length;

        return play(coins, 0, n - 1);
    }

    private static int play(int[] coins, int left, int right) {
        if (left > right) {
            return 0;
        }

        int pickLeft = coins[left]
                + Math.min(play(coins, left + 2, right), play(coins, left + 1, right - 1));
        int pickRight = coins[right]
                + Math.min(play(coins, left, right - 2), play(coins, left + 1, right - 1));

        int maxMoney = Math.max(pickLeft, pickRight);

        return maxMoney;
    }

    // S98.
    public static String shortestPath(String path) {
        String[] components = path.split("/");
        Stack<String> stack = new Stack<>();

        for (String component : components) {
            if (component.equals(".") || component.isEmpty()) {
                continue;
            } else if (component.equals("..")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                stack.push(component);
            }
        }

        StringBuilder sb = new StringBuilder("/");
        for (String dir : stack) {
            sb.append(dir).append("/");
        }

        return sb.toString();
    }

    // S99.
    public static String largestNumber(int[] nums) {
        String[] numStrings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            numStrings[i] = String.valueOf(nums[i]);
        }

        Arrays.sort(numStrings, new LargestNumberComparator());

        StringBuilder sb = new StringBuilder();
        for (String numString : numStrings) {
            sb.append(numString);
        }

        return sb.toString();
    }

    static class LargestNumberComparator implements Comparator<String> {
        @Override
        public int compare(String a, String b) {
            String order1 = a + b;
            String order2 = b + a;
            return order2.compareTo(order1);
        }
    }

    // S100.
    static class SnakesAndLadders {

        public int snakesAndLadders(int[] board) {
            int n = board.length;
            boolean[] visited = new boolean[n + 1];

            Queue<Integer> queue = new LinkedList<>();
            queue.offer(1);
            visited[1] = true;

            int turns = 0;

            while (!queue.isEmpty()) {
                int size = queue.size();

                for (int i = 0; i < size; i++) {
                    int square = queue.poll();

                    if (square == n - 1)
                        return turns;

                    for (int j = 1; j <= 6 && square + j < n; j++) {
                        int next = board[square + j] == -1 ? square + j : board[square + j];

                        if (!visited[next]) {
                            visited[next] = true;
                            queue.offer(next);
                        }
                    }
                }
                turns++;
            }
            return -1;
        }
    }

    // S101.
    public static int minTrialDrops(int eggs, int floors) {
        // Create a 2D array to store the minimum number of trial drops for each
        // subproblem
        int[][] dp = new int[eggs + 1][floors + 1];

        for (int i = 1; i <= eggs; i++) {
            dp[i][0] = 0;
            dp[i][1] = 1;
        }

        // If there is only one egg, we need to try dropping it from every floor
        for (int j = 1; j <= floors; j++) {
            dp[1][j] = j;
        }

        for (int i = 2; i <= eggs; i++) {
            for (int j = 2; j <= floors; j++) {
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = 1; k <= j; k++) {
                    int drops = 1 + Math.max(dp[i - 1][k - 1], dp[i][j - k]);
                    dp[i][j] = Math.min(dp[i][j], drops);
                }
            }
        }

        return dp[eggs][floors];
    }

    // S102.
    static class Point {
        double x;
        double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }

    public static boolean isInsidePolygon(Point[] polygon, Point p) {
        int n = polygon.length;
        int count = 0;

        for (int i = 0; i < n; i++) {
            Point a = polygon[i];
            Point b = polygon[(i + 1) % n];

            // Check if the ray intersects with the edge
            if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) {
                count++;
            }
        }

        // If the number of intersections is odd, point is inside the polygon
        return count % 2 != 0;
    }

    // S103.
    static class UnlockPatternCalculator {
        public int calculatePatterns(int n) {
            int[] path = new int[10];
            int count = 0;

            // Generate and count the patterns from 4 corners and 4 sides, and 1 center
            count += 4 * countPatterns(path, 1, n - 1);
            count += 4 * countPatterns(path, 2, n - 1);
            count += countPatterns(path, 5, n - 1);

            return count;
        }

        private int countPatterns(int[] path, int curr, int remaining) {
            if (remaining == 0)
                return 1;

            int count = 0;

            for (int i = 1; i <= 9; i++) {
                if (canVisit(path, curr, i)) {
                    path[i] = 1;
                    // Recursively count the patterns starting from the next key
                    count += countPatterns(path, i, remaining - 1);
                    path[i] = 0;
                }
            }

            return count;
        }

        private boolean canVisit(int[] path, int curr, int next) {
            if (path[next] != 0)
                return false;

            int currRow = (curr - 1) / 3;
            int currCol = (curr - 1) % 3;

            int nextRow = (next - 1) / 3;
            int nextCol = (next - 1) % 3;

            // If the next key is on the same row or column as the current key, return true
            if (currRow == nextRow || currCol == nextCol)
                return true;

            // If the third key between two keys is already visited, return true
            int mid = (curr + next) / 2;
            return path[mid] != 0;
        }
    }

    // S104.
    private static boolean isValidPartition(int[] nums, int k, int maxSum) {
        int sum = 0;
        int partitions = 1;

        for (int num : nums) {
            if (sum + num > maxSum) {
                sum = num;
                partitions++;

                if (partitions > k) {
                    return false;
                }
            } else {
                sum += num;
            }
        }

        return true;
    }

    public static int partitionArray(int[] nums, int k) {
        int maxNum = 0;
        int sumNum = 0;

        for (int num : nums) {
            maxNum = Math.max(maxNum, num);
            sumNum += num;
        }

        int left = maxNum;
        int right = sumNum;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (isValidPartition(nums, k, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }

    // S105.
    public static int minJumps(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return 0;
        }

        int[] jumps = new int[n];
        Arrays.fill(jumps, Integer.MAX_VALUE);
        jumps[0] = 0;

        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);

        while (!queue.isEmpty()) {
            int currentIndex = queue.poll();
            int currentJumps = jumps[currentIndex];

            int maxSteps = nums[currentIndex];
            for (int i = 1; i <= maxSteps; i++) {
                int nextIndex = currentIndex + i;
                if (nextIndex >= n) {
                    break;
                }

                if (currentJumps + 1 < jumps[nextIndex]) {
                    jumps[nextIndex] = currentJumps + 1;
                    queue.offer(nextIndex);
                }
            }
        }

        return jumps[n - 1];
    }

    // S106.
    public static boolean canFormCircle(List<String> words) {
        if (words == null || words.isEmpty()) {
            return false;
        }

        int n = words.size();
        boolean[] visited = new boolean[n];

        return dfs(words, words.size(), visited, words.get(0), words.get(0));
    }

    private static boolean dfs(List<String> words, int wordsSize, boolean[] visited, String startWord,
            String currentWord) {
        if (wordsSize == 1 && startWord.charAt(0) == currentWord.charAt(currentWord.length() - 1)) {
            return true;
        }

        for (int i = 0; i < words.size(); i++) {
            if (!visited[i] && currentWord.charAt(currentWord.length() - 1) == words.get(i).charAt(0)) {
                visited[i] = true;
                if (dfs(words, wordsSize - 1, visited, startWord, words.get(i))) {
                    return true;
                }
                visited[i] = false;
            }
        }

        return false;
    }

    // S107.
    public static Map<Character, Integer> solvePuzzle(String word1, String word2, String result) {
        Set<Character> letters = new HashSet<>();
        for (char c : (word1 + word2 + result).toCharArray()) {
            letters.add(c);
        }

        Map<Character, Integer> assignments = new HashMap<>();
        int[] digits = new int[10];
        boolean foundSolution = solveRecursively(word1, word2, result, letters, assignments, digits, 0);
        if (foundSolution) {
            return assignments;
        } else {
            return null;
        }
    }

    private static boolean solveRecursively(String word1, String word2, String result, Set<Character> letters,
            Map<Character, Integer> assignments, int[] digits, int index) {
        if (index == letters.size()) {
            return evaluateExpression(word1, word2, result, assignments);
        }

        char[] chars = new char[letters.size()];
        int i = 0;
        for (char c : letters) {
            chars[i++] = c;
        }

        for (int digit = (index == 0 ? 1 : 0); digit <= 9; digit++) {
            if (digits[digit] == 0) {
                if (digit == 0 && result.charAt(0) == chars[index]) {
                    continue; // Skip leading zero assignment if it matches the first character of the result
                }

                assignments.put(chars[index], digit);
                digits[digit] = 1;

                if (solveRecursively(word1, word2, result, letters, assignments, digits, index + 1)) {
                    return true;
                }

                digits[digit] = 0;
            }
        }

        return false;
    }

    private static boolean evaluateExpression(String word1, String word2, String result,
            Map<Character, Integer> assignments) {
        int num1 = getNumericValue(word1, assignments);
        int num2 = getNumericValue(word2, assignments);
        int res = getNumericValue(result, assignments);
        return (num1 + num2 == res);
    }

    private static int getNumericValue(String word, Map<Character, Integer> assignments) {
        int value = 0;
        for (char c : word.toCharArray()) {
            value = value * 10 + assignments.get(c);
        }
        return value;
    }

    // S108.
    // Solution in Q108.

    // S109.
    public static void printZigzag(String s, int k) {
        if (k == 1) {
            System.out.println(s);
            return;
        }

        StringBuilder[] rows = new StringBuilder[k];
        for (int i = 0; i < k; i++) {
            rows[i] = new StringBuilder();
        }

        int row = 0;
        boolean goingDown = true;
        boolean firstCh = true;

        for (char c : s.toCharArray()) {
            if (goingDown) {
                for (int i = 0; i < row; i++) {
                    rows[row].append(" ");
                }

                rows[row].append(c);

                for (int i = 0; i < k - row - 1; i++) {
                    rows[row].append(" ");
                }
            } else {
                for (int i = k - row - 1; i > 0; i--) {
                    rows[row].append(" ");
                }

                rows[row].append(c);

                for (int i = row; i > 0; i--) {
                    rows[row].append(" ");
                }
            }

            if (row == 0) {
                goingDown = true;
                if (!firstCh) {
                    for (int i = 0; i < k; i++) {
                        rows[row].append(" ");
                    }
                }
                firstCh = false;
            } else if (row == k - 1) {
                goingDown = false;
                for (int i = 0; i < k; i++) {
                    rows[row].append(" ");
                }
            }

            row += goingDown ? 1 : -1;
        }

        StringBuilder result = new StringBuilder();
        for (StringBuilder sb : rows) {
            result.append(sb).append("\n");
        }

        System.out.println(result);
    }

    // S110.
    public static TreeNode<Integer> convertToFullBinaryTree(TreeNode<Integer> root) {
        if (root == null) {
            return null;
        }

        if (root.left != null && root.right != null) {
            root.left = convertToFullBinaryTree(root.left);
            root.right = convertToFullBinaryTree(root.right);
        } else if (root.left != null) {
            root = convertToFullBinaryTree(root.left);
        } else if (root.right != null) {
            root = convertToFullBinaryTree(root.right);
        }

        return root;
    }

    // S111.
    public static ListNode rearrangeLinkedList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode current = head;
        ListNode temp = new ListNode(0);
        ListNode dummy = temp;
        boolean isLow = true;

        while (current != null && current.next != null) {
            if (isLow) {
                if (current.val < current.next.val) {
                    temp.next = current;
                    current = current.next;
                } else {
                    temp.next = current.next;
                    current.next = current.next.next;
                }
                temp = temp.next;
            } else {
                if (current.val > current.next.val) {
                    temp.next = current;
                    current = current.next;
                } else {
                    temp.next = current.next;
                    current.next = current.next.next;
                }
                temp = temp.next;
            }

            isLow = !isLow;
        }

        temp.next = current;

        return dummy.next;
    }

    private static void printLinkedList(ListNode head) {
        ListNode current = head;
        while (current != null) {
            System.out.print(current.val + " -> ");
            current = current.next;
        }
        System.out.println("null");
    }

    // S112.
    public static int[] reconstructArray(String[] clues) {
        int numOfMinus = 0;

        for (String clue : clues) {
            if (clue.equals("-")) {
                numOfMinus++;
            }
        }

        int N = clues.length;
        int[] result = new int[N];
        int j = 0;
        int number = numOfMinus;

        for (int i = 0; i < N; i++) {
            if (clues[i].equals("-")) {
                result[i] = j++;
            } else {
                result[i] = number++;
            }
        }

        return result;
    }

    // S113.
    // Solution in S84 Graph class member functions

    // S114.
    public static void checkSentences(String input) {
        String[] sentences = input.split("(?<=[.?!‽])\\s+");

        for (String sentence : sentences) {
            if (isValidSentence(sentence)) {
                System.out.println(sentence);
            }
        }
    }

    public static boolean isValidSentence(String sentence) {
        // Check if sentence starts with a capital letter,
        // and all other characters are lowercase letters, separators,
        // and sentence ends with a terminal mark immediately following a word
        if (!sentence.matches("^[A-Z][a-z\\s,:;.]*[a-z][.?!‽]$")) {
            return false;
        }

        String pattern = "\\s{2,}";
        Pattern regex = Pattern.compile(pattern);
        Matcher matcher = regex.matcher(sentence);
        // Check if there are two or more spaces between words
        if (matcher.find()) {
            return false;
        }

        return true;
    }

    // S115.
    public static boolean isPowerOfFour(int n) {
        // Check if n is a power of two
        if ((n & (n - 1)) != 0) {
            return false;
        }

        // Check if the only set bit is at an even position
        // n & 10101010 10101010 10101010 10101010
        if ((n & 0xAAAAAAAA) != 0) {
            return false;
        }

        return true;
    }

    // S116.
    static class NetworkNode {
        int id;
        int time;

        NetworkNode(int id, int time) {
            this.id = id;
            this.time = time;
        }
    }

    public static int propagateMessage(int N, int[][] edges) {
        List<List<NetworkNode>> graph = new ArrayList<>();
        for (int i = 0; i <= N; i++) {
            graph.add(new ArrayList<>());
        }

        for (int[] edge : edges) {
            int a = edge[0];
            int b = edge[1];
            int t = edge[2];
            graph.get(a).add(new NetworkNode(b, t));
        }

        int[] dist = new int[N + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[0] = 0;

        PriorityQueue<NetworkNode> pq = new PriorityQueue<>(Comparator.comparingInt(node -> node.time));
        pq.offer(new NetworkNode(0, 0));

        while (!pq.isEmpty()) {
            NetworkNode curr = pq.poll();
            if (curr.time > dist[curr.id]) {
                continue;
            }

            for (NetworkNode neighbor : graph.get(curr.id)) {
                int newTime = curr.time + neighbor.time;
                if (newTime < dist[neighbor.id]) {
                    dist[neighbor.id] = newTime;
                    pq.offer(new NetworkNode(neighbor.id, newTime));
                }
            }
        }

        int maxTime = 0;
        for (int i = 0; i <= N; i++) {
            maxTime = Math.max(maxTime, dist[i]);
        }

        return maxTime;
    }

    // S117.
    public static int throw_dice(int N, int faces, int total) {
        Map<String, Integer> memo = new HashMap<>();
        return countWays(N, faces, total, memo);
    }

    private static int countWays(int N, int faces, int total, Map<String, Integer> memo) {
        if (total < 0) {
            return 0;
        }
        if (N == 0) {
            return (total == 0) ? 1 : 0;
        }
        if (total == 0) {
            return 0;
        }

        String key = N + ":" + total;
        if (memo.containsKey(key)) {
            return memo.get(key);
        }

        int ways = 0;
        for (int face = 1; face <= faces; face++) {
            ways += countWays(N - 1, faces, total - face, memo);
        }

        memo.put(key, ways);

        return ways;
    }

    // S118.
    public static String nthTerm(int N) {
        if (N <= 0) {
            return "";
        }

        if (N == 1) {
            return "1";
        }

        String previousTerm = nthTerm(N - 1);
        StringBuilder currentTerm = new StringBuilder();

        char currentDigit = previousTerm.charAt(0);
        int count = 1;

        for (int i = 1; i < previousTerm.length(); i++) {
            char digit = previousTerm.charAt(i);

            if (digit == currentDigit) {
                count++;
            } else {
                currentTerm.append(count).append(currentDigit);
                currentDigit = digit;
                count = 1;
            }
        }

        currentTerm.append(count).append(currentDigit);

        return currentTerm.toString();
    }

    // S119.
    public static int stringMatch(String text, String pattern) {
        int n = text.length();
        int m = pattern.length();

        int[] prefixTable = buildPrefixTable(pattern);

        int i = 0;
        int j = 0;

        while (i < n) {
            if (text.charAt(i) == pattern.charAt(j)) {
                i++;
                j++;

                if (j == m) {
                    return i - j;
                }
            } else if (j > 0) {
                j = prefixTable[j - 1];
            } else {
                i++;
            }
        }

        return -1;
    }

    private static int[] buildPrefixTable(String pattern) {
        int m = pattern.length();
        int[] prefixTable = new int[m];
        int len = 0;

        int i = 1;
        while (i < m) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                len++;
                prefixTable[i] = len;
                i++;
            } else {
                if (len > 0) {
                    len = prefixTable[len - 1];
                } else {
                    prefixTable[i] = len;
                    i++;
                }
            }
        }

        return prefixTable;
    }

    // S120.
    public static int fewestBricks(List<List<Integer>> wall) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();

        for (List<Integer> row : wall) {
            int sum = 0;
            for (int i = 0; i < row.size() - 1; i++) {
                sum += row.get(i);
                frequencyMap.put(sum, frequencyMap.getOrDefault(sum, 0) + 1);
            }
        }

        int maxFrequency = 0;
        for (int frequency : frequencyMap.values()) {
            maxFrequency = Math.max(maxFrequency, frequency);
        }

        int rowCount = wall.size();
        return rowCount - maxFrequency;
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

        /*
         * Q71.
         * Given a positive integer n, find the smallest number of squared integers
         * which sum to n.
         * For example, given n = 13, return 2 since 13 = 3^2 + 2^2 = 9 + 4.
         * Given n = 27, return 3 since 27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9.
         */
        System.out.println("========= Q71 ==========");
        int nToFindSquaredInt = 13;
        int smallestSum = findSmallestSquaredSum(nToFindSquaredInt);
        System.out.println("Smallest number of squared integers for " + n + ": " + smallestSum);

        nToFindSquaredInt = 27;
        smallestSum = findSmallestSquaredSum(nToFindSquaredInt);
        System.out.println("Smallest number of squared integers for " + n + ": " + smallestSum);

        /*
         * Q72.
         * You are given an N by M matrix of 0s and 1s. Starting from the top left
         * corner, how many ways are there to reach the bottom right corner?
         * You can only move right and down. 0 represents an empty space while 1
         * represents a wall you cannot walk through.
         * For example, given the following matrix:
         * [[0, 0, 1],
         * [0, 0, 1],
         * [1, 0, 0]]
         * Return two, as there are only two ways to get to the bottom right:
         * Right, down, down, right
         * Down, right, down, right
         * The top left corner and bottom right corner will always be 0.
         */
        System.out.println("========= Q72 ==========");
        int[][] matrixOf01 = {
                { 0, 0, 1 },
                { 0, 0, 1 },
                { 1, 0, 0 }
        };
        int numPaths = countPaths(matrixOf01);
        System.out.println("Number of paths: " + numPaths);

        /*
         * Q73.
         * Given a list of words, return the shortest unique prefix of each word. For
         * example, given the list:
         * dog
         * cat
         * apple
         * apricot
         * fish
         * Return the list:
         * d
         * c
         * app
         * apr
         * f
         */
        System.out.println("========= Q73 ==========");
        String[] wordList = { "dog", "cat", "apple", "apricot", "fish" };
        List<String> prefixes = findShortestUniquePrefix(wordList);
        for (String prefix : prefixes) {
            System.out.println(prefix);
        }

        /*
         * Q74.
         * You are given an array of length n + 1 whose elements belong to the set {1,
         * 2, ..., n}. By the pigeonhole principle, there must be a duplicate. Find it
         * in linear time and space.
         */
        System.out.println("========= Q74 ==========");
        int[] arrWithDuplicate = { 1, 3, 4, 2, 2 };
        int duplicate = findDuplicate(arrWithDuplicate);
        System.out.println("Duplicate element: " + duplicate);

        /*
         * Q75.
         * Given an array of integers, return a new array where each element in the new
         * array is the number of smaller elements to the right of that element in the
         * original input array.
         * For example, given the array [3, 4, 9, 6, 1], return [1, 1, 2, 1, 0], since:
         * There is 1 smaller element to the right of 3
         * There is 1 smaller element to the right of 4
         * There are 2 smaller elements to the right of 9
         * There is 1 smaller element to the right of 6
         * There are no smaller elements to the right of 1
         */
        System.out.println("========= Q75 ==========");
        int[] numsToFindSmallerRightElements = { 3, 4, 9, 6, 1 };
        int[] numsOfSmallerRightElements = countSmallerElements(numsToFindSmallerRightElements);

        System.out.println(Arrays.toString(numsOfSmallerRightElements));

        /*
         * Q76.
         * Implement a 2D iterator class. It will be initialized with an array of
         * arrays, and should implement the following methods:
         * next(): returns the next element in the array of arrays. If there are no more
         * elements, raise an exception.
         * has_next(): returns whether or not the iterator still has elements left.
         * For example, given the input [[1, 2], [3], [], [4, 5, 6]], calling next()
         * repeatedly should output 1, 2, 3, 4, 5, 6.
         * Do not use flatten or otherwise clone the arrays. Some of the arrays can be
         * empty.
         */
        System.out.println("========= Q76 ==========");
        List<List<Integer>> arrays = List.of(
                List.of(1, 2),
                List.of(3),
                List.of(),
                List.of(4, 5, 6));

        TwoDIterator<Integer> twoDIterator = new TwoDIterator<>(arrays);

        while (twoDIterator.hasNext()) {
            System.out.println(twoDIterator.next());
        }

        /*
         * Q77.
         * Given an N by N matrix, rotate it by 90 degrees clockwise.
         * For example, given the following matrix:
         * [[1, 2, 3],
         * [4, 5, 6],
         * [7, 8, 9]]
         * you should return:
         * [[7, 4, 1],
         * [8, 5, 2],
         * [9, 6, 3]]
         * Follow-up: What if you couldn't use any extra space?
         */
        System.out.println("========= Q77 ==========");
        int[][] matrixToRotate = {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 }
        };

        rotateMatrix(matrixToRotate);

        for (int[] row : matrixToRotate) {
            for (int num : row) {
                System.out.print(num + " ");
            }
            System.out.println();
        }

        /*
         * Q78.
         * Given a linked list, sort it in O(n log n) time and constant space.
         * For example, the linked list 4 -> 1 -> -3 -> 99 should become -3 -> 1 -> 4 ->
         * 99.
         */
        System.out.println("========= Q78 ==========");
        ListNode listToSort = new ListNode(4);
        listToSort.next = new ListNode(1);
        listToSort.next.next = new ListNode(-3);
        listToSort.next.next.next = new ListNode(99);

        ListNode sortedList = sortList(listToSort);

        while (sortedList != null) {
            System.out.print(sortedList.val + " ");
            sortedList = sortedList.next;
        }
        System.out.println();

        /*
         * Q79.
         * Given a start word, an end word, and a dictionary of valid words, find the
         * shortest transformation sequence from start to end such that only one letter
         * is changed at each step of the sequence, and each transformed word exists in
         * the dictionary. If there is no possible transformation, return null. Each
         * word in the dictionary have the same length as start and end and is
         * lowercase.
         * For example, given start = "dog", end = "cat", and dictionary = {"dot",
         * "dop", "dat", "cat"}, return ["dog", "dot", "dat", "cat"].
         * Given start = "dog", end = "cat", and dictionary = {"dot", "tod", "dat",
         * "dar"}, return null as there is no possible transformation from dog to cat.
         */
        System.out.println("========= Q79 ==========");
        String start = "dog";
        String end = "cat";
        Set<String> dictionary1 = new HashSet<>(Arrays.asList("dot", "dop", "dat", "cat"));

        WordTransformation transformer = new WordTransformation();
        List<String> transformation = transformer.findTransformation(start, end, dictionary1);

        if (transformation != null) {
            System.out.println(transformation);
        } else {
            System.out.println("No transformation sequence found.");
        }

        Set<String> dictionary2 = new HashSet<>(Arrays.asList("dot", "tod", "dat", "dar"));
        transformation = transformer.findTransformation(start, end, dictionary2);

        if (transformation != null) {
            System.out.println(transformation);
        } else {
            System.out.println("No transformation sequence found.");
        }

        /*
         * Q80.
         * Given a string s and a list of words words, where each word is the same
         * length, find all starting indices of substrings in s that is a concatenation
         * of every word in words exactly once.
         * For example, given s = "dogcatcatcodecatdog" and words = ["cat", "dog"],
         * return [0, 13], since "dogcat" starts at index 0 and "catdog" starts at index
         * 13.
         * Given s = "barfoobazbitbyte" and words = ["dog", "cat"], return [] since
         * there are no substrings composed of "dog" and "cat" in s.
         * The order of the indices does not matter.
         */
        System.out.println("========= Q80 ==========");
        String s1 = "dogcatcatcodecatdog";
        String[] words1 = { "cat", "dog" };
        List<Integer> substringResult1 = findSubstring(s1, words1);
        System.out.println(substringResult1); // Output: [0, 13]

        String s2 = "barfoobazbitbyte";
        String[] words2 = { "dog", "cat" };
        List<Integer> substringResult2 = findSubstring(s2, words2);
        System.out.println(substringResult2); // Output: []

        /*
         * Q81.
         * Describe and give an example of each of the following types of polymorphism:
         * Ad-hoc polymorphism
         * Parametric polymorphism
         * Subtype polymorphism
         */
        System.out.println("========= Q81 ==========");
        // Solution implemented in S81

        /*
         * Q82.
         * Given the sequence of keys visited by a postorder traversal of a binary
         * search tree, reconstruct the tree.
         * For example, given the sequence 2, 4, 3, 8, 7, 5, you should construct the
         * following tree:
         * "    5      "
         * "   / \     "
         * "  3   7    "
         * " / \   \   "
         * "2   4   8  "
         */
        System.out.println("========= Q82 ==========");
        int[] postorder = { 2, 4, 3, 8, 7, 5 };

        TreeNode<Integer> treeRootFromPostorder = buildTreeFromPostorder(postorder);

        System.out.println("Inorder traversal:");
        inorderTraversal(treeRootFromPostorder);
        System.out.println();

        /*
         * Q83.
         * Given a stack of N elements, interleave the first half of the stack with the
         * second half reversed using only one other queue. This should be done
         * in-place.
         * Recall that you can only push or pop from a stack, and enqueue or dequeue
         * from a queue.
         * For example, if the stack is [1, 2, 3, 4, 5], it should become [1, 5, 2, 4,
         * 3]. If the stack is [1, 2, 3, 4], it should become [1, 4, 2, 3].
         * Hint: Try working backwards from the end state.
         */
        System.out.println("========= Q83 ==========");
        Stack<Integer> stack1 = new Stack<>();
        stack1.push(1);
        stack1.push(2);
        stack1.push(3);
        stack1.push(4);
        stack1.push(5);
        interleaveStack(stack1);
        System.out.println(stack1); // Output: [1, 5, 2, 4, 3]

        Stack<Integer> stack2 = new Stack<>();
        stack2.push(1);
        stack2.push(2);
        stack2.push(3);
        stack2.push(4);
        interleaveStack(stack2);
        System.out.println(stack2); // Output: [1, 4, 2, 3]

        /*
         * Q84.
         * A graph is minimally-connected if it is connected and there is no edge that
         * can be removed while still leaving the graph connected. For example, any
         * binary tree is minimally-connected.
         * Given an undirected graph, check if the graph is minimally-connected. You can
         * choose to represent the graph as either an adjacency matrix or adjacency
         * list.
         */
        System.out.println("========= Q84 ==========");
        Graph minimalGraph = new Graph(5);
        minimalGraph.addEdge(0, 1);
        minimalGraph.addEdge(0, 2);
        minimalGraph.addEdge(2, 3);
        minimalGraph.addEdge(2, 4);

        boolean isMinimallyConnected = minimalGraph.isMinimallyConnected();
        System.out.println("Is the graph minimally-connected? " + isMinimallyConnected);

        /*
         * Q85.
         * What will this code print out?
         * "def make_functions():          "
         * "    flist = []                 "
         * "                               "
         * "    for i in [1, 2, 3]:        "
         * "        def print_i():         "
         * "            print(i)           "
         * "        flist.append(print_i)  "
         * "                               "
         * "    return flist               "
         * "                               "
         * "functions = make_functions()   "
         * "for f in functions:            "
         * "    f()                        "
         * How can we make it print out what we apparently want?
         */
        /*
         * It will print '3' three times since make_functions() refers to 'i' in its
         * enclosing scope. It should be corrected by using default argument x=i in
         * print_i as below.
         * "def make_functions():          "
         * "    flist = []                 "
         * "                               "
         * "    for i in [1, 2, 3]:        "
         * "        def print_i(x=i):      "
         * "            print(i)           "
         * "        flist.append(print_i)  "
         * "                               "
         * "    return flist               "
         * "                               "
         * "functions = make_functions()   "
         * "for f in functions:            "
         * "    f()                        "
         */
        System.out.println("========= Q85 ==========");

        /*
         * Q86.
         * Given a circular array, compute its maximum subarray sum in O(n) time. A
         * subarray can be empty, and in this case the sum is 0.
         * For example, given [8, -1, 3, 4], return 15 as we choose the numbers 3, 4,
         * and 8 where the 8 is obtained from wrapping around.
         * Given [-4, 5, 1, 0], return 6 as we choose the numbers 5 and 1.
         */
        System.out.println("========= Q86 ==========");
        int[] circularArr1 = { 8, -1, 3, 4 };
        System.out.println(getMaxSubarraySum(circularArr1)); // Output: 15

        int[] circularArr2 = { -4, 5, 1, 0 };
        System.out.println(getMaxSubarraySum(circularArr2)); // Output: 6

        /*
         * Q87.
         * You are given an array of nonnegative integers. Let's say you start at the
         * beginning of the array and are trying to advance to the end. You can advance
         * at most, the number of steps that you're currently on. Determine whether you
         * can get to the end of the array.
         * For example, given the array [1, 3, 1, 2, 0, 1], we can go from indices 0 ->
         * 1 -> 3 -> 5, so return true.
         * Given the array [1, 2, 1, 0, 0], we can't reach the end, so return false.
         */
        System.out.println("========= Q87 ==========");
        int[] stepsArr1 = { 1, 3, 1, 2, 0, 1 };
        System.out.println(canReachEnd(stepsArr1)); // Output: true

        int[] stepsArr2 = { 1, 2, 1, 0, 0 };
        System.out.println(canReachEnd(stepsArr2)); // Output: false

        /*
         * Q88.
         * Given a set of distinct positive integers, find the largest subset such that
         * every pair of elements in the subset (i, j) satisfies either i % j = 0 or j %
         * i = 0.
         * For example, given the set [3, 5, 10, 20, 21], you should return [5, 10, 20].
         * Given [1, 3, 6, 24], return [1, 3, 6, 24].
         */
        System.out.println("========= Q88 ==========");
        int[] numsArr1 = { 3, 5, 10, 20, 21 };
        System.out.println(largestDivisibleSubset(numsArr1)); // Output: [5, 10, 20]

        int[] numsArr2 = { 1, 3, 6, 24 };
        System.out.println(largestDivisibleSubset(numsArr2)); // Output: [1, 3, 6, 24]

        /*
         * Q89.
         * Suppose an array sorted in ascending order is rotated at some pivot unknown
         * to you beforehand. Find the minimum element in O(log N) time. You may assume
         * the array does not contain duplicates.
         * For example, given [5, 7, 10, 3, 4], return 3.
         */
        System.out.println("========= Q89 ==========");
        int[] numsToFindPivot = { 5, 7, 10, 3, 4 };
        System.out.println(findMin(numsToFindPivot)); // Output: 3

        /*
         * Q90.
         * Given an undirected graph G, check whether it is bipartite. Recall that a
         * graph is bipartite if its vertices can be divided into two independent sets,
         * U and V, such that no edge connects vertices of the same set.
         */
        System.out.println("========= Q90 ==========");
        Graph bipartiteGraph = new Graph(4);
        bipartiteGraph.addEdge(0, 1);
        bipartiteGraph.addEdge(1, 2);
        bipartiteGraph.addEdge(2, 3);
        bipartiteGraph.addEdge(3, 0);

        boolean isBipartite = bipartiteGraph.isBipartite();
        System.out.println("Is the graph bipartite? " + isBipartite);

        /*
         * Q91.
         * Given a linked list of numbers and a pivot k, partition the linked list so
         * that all nodes less than k come before nodes greater than or equal to k.
         * For example, given the linked list 5 -> 1 -> 8 -> 0 -> 3 and k = 3, the
         * solution could be 1 -> 0 -> 5 -> 8 -> 3.
         */
        System.out.println("========= Q91 ==========");
        ListNode headToPartition = new ListNode(5);
        headToPartition.next = new ListNode(1);
        headToPartition.next.next = new ListNode(8);
        headToPartition.next.next.next = new ListNode(0);
        headToPartition.next.next.next.next = new ListNode(3);

        int kToPartition = 3;

        ListNode newList = partition(headToPartition, kToPartition);

        while (newList != null) {
            System.out.print(newList.val + " ");
            newList = newList.next;
        }
        System.out.println();

        /*
         * Q92.
         * Given a string and a pattern, find the starting indices of all occurrences of
         * the pattern in the string. For example, given the string "abracadabra" and
         * the pattern "abr", you should return [0, 7].
         */
        System.out.println("========= Q92 ==========");
        String strToFindPattern = "abracadabra";
        String pattern = "abr";

        List<Integer> indices = findPatternIndices(strToFindPattern, pattern);

        System.out.println("Indices of pattern occurrences: " + indices);

        /*
         * Q93.
         * Given a string of digits, generate all possible valid IP address
         * combinations.
         * IP addresses must follow the format A.B.C.D, where A, B, C, and D are numbers
         * between 0 and 255. Zero-prefixed numbers, such as 01 and 065, are not
         * allowed, except for 0 itself.
         * For example, given "2542540123", you should return ['254.25.40.123',
         * '254.254.0.123'].
         */
        System.out.println("========= Q93 ==========");
        String ipAddressString = "2542540123";
        List<String> ipAddressList = generateIPAddresses(ipAddressString);

        System.out.println("Valid IP addresses: " + ipAddressList);

        /*
         * Q94.
         * The horizontal distance of a binary tree node describes how far left or right
         * the node will be when the tree is printed out.
         * More rigorously, we can define it as follows:
         * The horizontal distance of the root is 0.
         * The horizontal distance of a left child is hd(parent) - 1.
         * The horizontal distance of a right child is hd(parent) + 1.
         * For example, for the following tree, hd(1) = -2, and hd(6) = 0.
         * "             5             "
         * "          /     \          "
         * "        3         7        "
         * "      /  \      /   \      "
         * "    1     4    6     9     "
         * "   /                /      "
         * "  0                8       "
         * The bottom view of a tree, then, consists of the lowest node at each
         * horizontal distance. If there are two nodes at the same depth and horizontal
         * distance, either is acceptable.
         * For this tree, for example, the bottom view could be [0, 1, 3, 6, 8, 9].
         * Given the root to a binary tree, return its bottom view.
         */
        System.out.println("========= Q94 ==========");
        NodeWithHorizontalDistance treeRootWithHorizontalDistance = new NodeWithHorizontalDistance(5);
        treeRootWithHorizontalDistance.left = new NodeWithHorizontalDistance(3);
        treeRootWithHorizontalDistance.right = new NodeWithHorizontalDistance(7);
        treeRootWithHorizontalDistance.left.left = new NodeWithHorizontalDistance(1);
        treeRootWithHorizontalDistance.left.right = new NodeWithHorizontalDistance(4);
        treeRootWithHorizontalDistance.right.left = new NodeWithHorizontalDistance(6);
        treeRootWithHorizontalDistance.right.right = new NodeWithHorizontalDistance(9);
        treeRootWithHorizontalDistance.left.left.left = new NodeWithHorizontalDistance(0);
        treeRootWithHorizontalDistance.right.right.left = new NodeWithHorizontalDistance(8);

        List<Integer> bottomView = bottomView(treeRootWithHorizontalDistance);

        System.out.println(bottomView); // Output: [0, 1, 3, 6, 8, 9]

        /*
         * Q95.
         * https://en.wikipedia.org/wiki/Roman_numerals
         * Given a number in Roman numeral format, convert it to decimal.
         * The values of Roman numerals are as follows:
         * " {              "
         * "     'M': 1000, "
         * "     'D': 500,  "
         * "     'C': 100,  "
         * "     'L': 50,   "
         * "     'X': 10,   "
         * "     'V': 5,    "
         * "     'I': 1     "
         * " }              "
         * In addition, note that the Roman numeral system uses subtractive notation for
         * numbers such as IV and XL.
         * https://en.wikipedia.org/wiki/Subtractive_notation
         * For the input XIV, for instance, you should return 14.
         */
        System.out.println("========= Q95 ==========");
        int decimalValue = romanToDecimal("XIV");
        System.out.println(decimalValue); // Output: 14

        /*
         * Q96.
         * Write an algorithm that computes the reversal of a directed graph. For
         * example, if a graph consists of A -> B -> C, it should become A <- B <- C.
         */
        System.out.println("========= Q96 ==========");
        ReversingGraph originalGraph = new ReversingGraph();
        originalGraph.addEdge("A", "B");
        originalGraph.addEdge("B", "C");
        originalGraph.addEdge("C", "D");

        System.out.println("Original graph:");
        originalGraph.printGraph();

        ReversingGraph reversedGraph = originalGraph.reverse();

        System.out.println("Reversed graph:");
        reversedGraph.printGraph();

        /*
         * Q97.
         * In front of you is a row of N coins, with values v1, v1, ..., vn.
         * You are asked to play the following game. You and an opponent take turns
         * choosing either the first or last coin from the row, removing it from the
         * row, and receiving the value of the coin.
         * Write a program that returns the maximum amount of money you can win with
         * certainty, if you move first, assuming your opponent plays optimally.
         */
        System.out.println("========= Q97 ==========");
        int[] coins = { 3, 9, 1, 2 };
        int maxMoney = maxMoney(coins);
        System.out.println("Maximum amount of money: " + maxMoney);

        /*
         * Q98.
         * Given an absolute pathname that may have . or .. as part of it, return the
         * shortest standardized path.
         * For example, given "/usr/bin/../bin/./scripts/../", return "/usr/bin/".
         */
        System.out.println("========= Q98 ==========");
        String path = "/usr/bin/../bin/./scripts/../";
        String standardizedPath = shortestPath(path);
        System.out.println("Shortest standardized path: " + standardizedPath);

        /*
         * Q99.
         * Given a list of numbers, create an algorithm that arranges them in order to
         * form the largest possible integer. For example, given [10, 7, 76, 415], you
         * should return 77641510.
         */
        System.out.println("========= Q99 ==========");
        int[] numsToFormLargest = { 10, 7, 76, 415 };
        String largestNum = largestNumber(numsToFormLargest);
        System.out.println("Largest number formation: " + largestNum);

        /*
         * Q100.
         * https://en.wikipedia.org/wiki/Snakes_and_Ladders
         * Snakes and Ladders is a game played on a 10 x 10 board, the goal of which is
         * get from square 1 to square 100. On each turn players will roll a six-sided
         * die and move forward a number of spaces equal to the result. If they land on
         * a square that represents a snake or ladder, they will be transported ahead or
         * behind, respectively, to a new square.
         * Find the smallest number of turns it takes to play snakes and ladders.
         * For convenience, here are the squares representing snakes and ladders, and
         * their outcomes:
         * snakes = {16: 6, 48: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95:
         * 75, 98: 78}
         * ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80:
         * 100}
         */
        System.out.println("========= Q100 ==========");
        SnakesAndLadders snakesAndLadders = new SnakesAndLadders();

        int[] board = new int[101];
        Arrays.fill(board, -1);

        Map<Integer, Integer> snakes = new HashMap<>();
        snakes.put(16, 6);
        snakes.put(48, 26);
        snakes.put(49, 11);
        snakes.put(56, 53);
        snakes.put(62, 19);
        snakes.put(64, 60);
        snakes.put(87, 24);
        snakes.put(93, 73);
        snakes.put(95, 75);
        snakes.put(98, 78);

        Map<Integer, Integer> ladders = new HashMap<>();
        ladders.put(1, 38);
        ladders.put(4, 14);
        ladders.put(9, 31);
        ladders.put(21, 42);
        ladders.put(28, 84);
        ladders.put(36, 44);
        ladders.put(51, 67);
        ladders.put(71, 91);
        ladders.put(80, 100);

        for (Map.Entry<Integer, Integer> entry : snakes.entrySet()) {
            board[entry.getKey()] = entry.getValue();
        }

        for (Map.Entry<Integer, Integer> entry : ladders.entrySet()) {
            board[entry.getKey()] = entry.getValue();
        }

        int minTurns = snakesAndLadders.snakesAndLadders(board);
        System.out.println("Minimum turns to win the game: " + minTurns);

        /*
         * Q101.
         * You are given N identical eggs and access to a building with k floors. Your
         * task is to find the lowest floor that will cause an egg to break, if dropped
         * from that floor. Once an egg breaks, it cannot be dropped again. If an egg
         * breaks when dropped from the xth floor, you can assume it will also break
         * when dropped from any floor greater than x.
         * Write an algorithm that finds the minimum number of trial drops it will take,
         * in the worst case, to identify this floor.
         * For example, if N = 1 and k = 5, we will need to try dropping the egg at
         * every floor, beginning with the first, until we reach the fifth floor, so our
         * solution will be 5.
         */
        System.out.println("========= Q101 ==========");
        int eggs = 1;
        int floors = 5;
        int minTrials = minTrialDrops(eggs, floors);
        System.out.println("The minimum number of trial drops required is: " + minTrials);

        /*
         * Q102.
         * You are given a list of N points (x1, y1), (x2, y2), ..., (xN, yN)
         * representing a polygon. You can assume these points are given in order; that
         * is, you can construct the polygon by connecting point 1 to point 2, point 2
         * to point 3, and so on, finally looping around to connect point N to point 1.
         * Determine if a new point p lies inside this polygon. (If p is on the boundary
         * of the polygon, you should return False).
         */
        System.out.println("========= Q102 ==========");
        Point[] polygon = { new Point(0, 0), new Point(4, 0), new Point(4, 4), new Point(0, 4) };
        Point p1 = new Point(2, 2);

        boolean isInside = isInsidePolygon(polygon, p1);
        System.out.println("Point is inside polygon: " + isInside);

        Point p2 = new Point(2, 4.1);
        isInside = isInsidePolygon(polygon, p2);
        System.out.println("Point is inside polygon: " + isInside);

        /*
         * Q103.
         * One way to unlock an Android phone is through a pattern of swipes across a
         * 1-9 keypad.
         * For a pattern to be valid, it must satisfy the following:
         * All of its keys must be distinct.
         * It must not connect two keys by jumping over a third key, unless that key has
         * already been used.
         * For example, 4 - 2 - 1 - 7 is a valid pattern, whereas 2 - 1 - 7 is not.
         * Find the total number of valid unlock patterns of length N, where 1 <= N <=
         * 9.
         */
        System.out.println("========= Q103 ==========");
        UnlockPatternCalculator calculator = new UnlockPatternCalculator();
        int numOfSwipes = 4;

        int totalPatterns = calculator.calculatePatterns(numOfSwipes);
        System.out.println("Total number of valid unlock patterns: " + totalPatterns);

        /*
         * Q104.
         * Given an array of numbers N and an integer k, your task is to split N into k
         * partitions such that the maximum sum of any partition is minimized. Return
         * this sum.
         * For example, given N = [5, 1, 2, 7, 3, 4] and k = 3, you should return 8,
         * since the optimal partition is [5, 1, 2], [7], [3, 4].
         */
        System.out.println("========= Q104 ==========");
        int[] numsToFindMinimizedMaxPartition = { 5, 1, 2, 7, 3, 4 };
        int numPartition = 3;

        int minMaxSum = partitionArray(numsToFindMinimizedMaxPartition, numPartition);
        System.out.println("Minimum maximum sum of partitions: " + minMaxSum);

        /*
         * Q105.
         * You are given an array of integers, where each element represents the maximum
         * number of steps that can be jumped going forward from that element. Write a
         * function to return the minimum number of jumps you must take in order to get
         * from the start to the end of the array.
         * For example, given [6, 2, 4, 0, 5, 1, 1, 4, 2, 9], you should return 2, as
         * the optimal solution involves jumping from 6 to 5, and then from 5 to 9.
         */
        System.out.println("========= Q105 ==========");
        int[] stepsArr = { 6, 2, 4, 0, 5, 1, 1, 4, 2, 9 };
        int minJumps = minJumps(stepsArr);
        System.out.println("Minimum number of jumps: " + minJumps);

        /*
         * Q106.
         * Given a list of words, determine whether the words can be chained to form a
         * circle. A word X can be placed in front of another word Y in a circle if the
         * last character of X is same as the first character of Y.
         * For example, the words ['chair', 'height', 'racket', touch', 'tunic'] can
         * form the following circle: chair --> racket --> touch --> height --> tunic
         * --> chair.
         */
        System.out.println("========= Q106 ==========");
        List<String> listOfWords = new ArrayList<>();
        listOfWords.add("chair");
        listOfWords.add("height");
        listOfWords.add("racket");
        listOfWords.add("touch");
        listOfWords.add("tunic");

        boolean canFormCircle = canFormCircle(listOfWords);
        System.out.println("Can form a circle: " + canFormCircle);

        /*
         * Q107.
         * A cryptarithmetic puzzle is a mathematical game where the digits of some
         * numbers are represented by letters. Each letter represents a unique digit.
         * For example, a puzzle of the form:
         * "   SEND    "
         * " + MORE    "
         * "--------   "
         * " MONEY     "
         * may have the solution:
         * {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O', 0, 'R': 8, 'Y': 2}
         * Given a three-word puzzle like the one above, create an algorithm that finds
         * a solution.
         */
        System.out.println("========= Q107 ==========");
        String word1 = "SEND";
        String word2 = "MORE";
        String wordsMathResult = "MONEY";

        Map<Character, Integer> solution = solvePuzzle(word1, word2, wordsMathResult);

        if (solution != null) {
            System.out.println("Solution found:");
            for (Map.Entry<Character, Integer> entry : solution.entrySet()) {
                System.out.println(entry.getKey() + ": " + entry.getValue());
            }
        } else {
            System.out.println("No solution found.");
        }

        /*
         * Q108.
         * Given an array of a million integers between zero and a billion, out of
         * order, how can you efficiently sort it? Assume that you cannot store an array
         * of a billion elements in memory.
         */
        System.out.println("========= Q108 ==========");
        /*
         * High-level approach:
         * 1. Divide the array into smaller chunks that can fit into memory.
         * 2. Sort each chunk individually using an efficient in-memory sorting such as
         * quicksort or mergesort.
         * 3. Write each sorted chunk to a single file. Structure data in the file using
         * priority queue or min-heap.
         * 4. Perform a k-way merge sort on the sorted files. Repeatedly select the
         * smallest element from each file and write it to the output file.
         */

        /*
         * Q109.
         * Given a string and a number of lines k, print the string in zigzag form. In
         * zigzag, characters are printed out diagonally from top left to bottom right
         * until reaching the kth line, then back up to top right, and so on.
         * For example, given the sentence "thisisazigzag" and k = 4, you should print:
         * " t     a     g "
         * "  h   s z   a  "
         * "   i i   i z   "
         * "    s     g    "
         */
        System.out.println("========= Q109 ==========");
        String sentence = "thisisazigzag";
        int numLines = 4;
        printZigzag(sentence, numLines);

        /*
         * Q110.
         * Recall that a full binary tree is one in which each node is either a leaf
         * node, or has two children. Given a binary tree, convert it to a full one by
         * removing nodes with only one child.
         * For example, given the following tree:
         * "          0                "
         * "       /     \             "
         * "     1         2           "
         * "   /            \          "
         * " 3                 4       "
         * "   \             /   \     "
         * "     5          6     7    "
         * You should convert it to:
         * "     0             "
         * "   /     \         "
         * " 5         4       "
         * "         /   \     "
         * "        6     7    "
         */
        System.out.println("========= Q110 ==========");
        TreeNode<Integer> treeToConvert = new TreeNode<>(0);
        treeToConvert.left = new TreeNode<>(1);
        treeToConvert.right = new TreeNode<>(2);
        treeToConvert.left.left = new TreeNode<>(3);
        treeToConvert.left.left.right = new TreeNode<>(5);
        treeToConvert.right.right = new TreeNode<>(4);
        treeToConvert.right.right.left = new TreeNode<>(6);
        treeToConvert.right.right.right = new TreeNode<>(7);

        System.out.println("Original Binary Tree:");
        printInorder(treeToConvert);
        System.out.println();

        TreeNode<Integer> fullBinaryTree = convertToFullBinaryTree(treeToConvert);

        System.out.println("Full Binary Tree:");
        printInorder(fullBinaryTree);
        System.out.println();

        /*
         * Q111.
         * Given a linked list, rearrange the node values such that they appear in
         * alternating low -> high -> low -> high ... form. For example, given 1 -> 2 ->
         * 3 -> 4 -> 5, you should return 1 -> 3 -> 2 -> 5 -> 4.
         */
        System.out.println("========= Q111 ==========");
        ListNode listToRearrange = new ListNode(1);
        listToRearrange.next = new ListNode(2);
        listToRearrange.next.next = new ListNode(3);
        listToRearrange.next.next.next = new ListNode(4);
        listToRearrange.next.next.next.next = new ListNode(5);

        ListNode rearrangedList = rearrangeLinkedList(listToRearrange);

        printLinkedList(rearrangedList);

        /*
         * Q112.
         * The sequence [0, 1, ..., N] has been jumbled, and the only clue you have for
         * its order is an array representing whether each number is larger or smaller
         * than the last. Given this information, reconstruct an array that is
         * consistent with it. For example, given [None, +, +, -, +], you could return
         * [1, 2, 3, 0, 4].
         */
        System.out.println("========= Q112 ==========");
        String[] clues = { "None", "+", "+", "-", "+" };
        int[] reconstructedArray = reconstructArray(clues);

        System.out.println("Reconstructed Array:");
        for (int i = 0; i < reconstructedArray.length; i++) {
            System.out.print(reconstructedArray[i] + " ");
        }
        System.out.println();

        /*
         * Q113.
         * A bridge in a connected (undirected) graph is an edge that, if removed,
         * causes the graph to become disconnected. Find all the bridges in a graph.
         */
        System.out.println("========= Q113 ==========");
        int V = 5;
        Graph bridgeGraph = new Graph(V);
        bridgeGraph.addEdge(0, 1);
        bridgeGraph.addEdge(1, 2);
        bridgeGraph.addEdge(2, 0);
        bridgeGraph.addEdge(1, 3);
        bridgeGraph.addEdge(3, 4);

        List<int[]> bridges = bridgeGraph.findBridges();
        for (int[] bridge : bridges) {
            System.out.println(bridge[0] + " - " + bridge[1]);
        }

        /*
         * Q114.
         * Create a basic sentence checker that takes in a stream of characters and
         * determines whether they form valid sentences. If a sentence is valid, the
         * program should print it out.
         * We can consider a sentence valid if it conforms to the following rules:
         * The sentence must start with a capital letter, followed by a lowercase letter
         * or a space.
         * All other characters must be lowercase letters, separators (,,;,:) or
         * terminal marks (.,?,!,‽).
         * There must be a single space between each word.
         * The sentence must end with a terminal mark immediately following a word.
         */
        System.out.println("========= Q114 ==========");
        String sentenceToCheck = "This is a valid sentence. Another valid sentence? No? Invalid! Two   more spaces. This is, ,an invalid sentence";
        checkSentences(sentenceToCheck);

        /*
         * Q115.
         * Given a 32-bit positive integer N, determine whether it is a power of four in
         * faster than O(log N) time.
         */
        System.out.println("========= Q115 ==========");
        int powerOfFour = 256; // 00000000 00000000 00000001 00000000
        boolean isPowerOfFour = isPowerOfFour(powerOfFour);
        System.out.println(powerOfFour + " is a power of four: " + isPowerOfFour);

        int notPowerOfFour = 128; // 00000000 00000000 00000000 10000000
        isPowerOfFour = isPowerOfFour(notPowerOfFour);
        System.out.println(notPowerOfFour + " is a power of four: " + isPowerOfFour);

        /*
         * Q116.
         * A network consists of nodes labeled 0 to N. You are given a list of edges (a,
         * b, t), describing the time t it takes for a message to be sent from node a to
         * node b. Whenever a node receives a message, it immediately passes the message
         * on to a neighboring node, if possible.
         * Assuming all nodes are connected, determine how long it will take for every
         * node to receive a message that begins at node 0.
         * For example, given N = 5, and the following edges:
         * " edges = [         "
         * "     (0, 1, 5),    "
         * "     (0, 2, 3),    "
         * "     (0, 5, 4),    "
         * "     (1, 3, 8),    "
         * "     (2, 3, 1),    "
         * "     (3, 5, 10),   "
         * "     (3, 4, 5)     "
         * " ]                 "
         * You should return 9, because propagating the message from 0 -> 2 -> 3 -> 4
         * will take that much time.
         */
        System.out.println("========= Q116 ==========");
        int numNodes = 5;
        int[][] edges = {
                { 0, 1, 5 },
                { 0, 2, 3 },
                { 0, 5, 4 },
                { 1, 3, 8 },
                { 2, 3, 1 },
                { 3, 5, 10 },
                { 3, 4, 5 }
        };

        int leadTime = propagateMessage(numNodes, edges);
        System.out.println("Time taken for every node to receive the message: " + leadTime);

        /*
         * Q117.
         * Write a function, throw_dice(N, faces, total), that determines how many ways
         * it is possible to throw N dice with some number of faces each to get a
         * specific total.
         * For example, throw_dice(3, 6, 7) should equal 15.
         */
        System.out.println("========= Q117 ==========");
        int numDices = 3;
        int faces = 6;
        int total = 7;
        int waysToGetTarget = throw_dice(numDices, faces, total);
        System.out.println("Number of ways: " + waysToGetTarget);

        /*
         * Q118.
         * The "look and say" sequence is defined as follows: beginning with the term 1,
         * each subsequent term visually describes the digits appearing in the previous
         * term. The first few terms are as follows:
         * 1
         * 11
         * 21
         * 1211
         * 111221
         * As an example, the fourth term is 1211, since the third term consists of one
         * 2 and one 1.
         * Given an integer N, print the Nth term of this sequence.
         */
        System.out.println("========= Q118 ==========");
        int term = 5;
        String nthTerm = nthTerm(term);
        System.out.println("Nth term of the 'look and say' sequence for N = " + term + ": " + nthTerm);

        /*
         * Q119.
         * Implement an efficient string matching algorithm.
         * That is, given a string of length N and a pattern of length k, write a
         * program that searches for the pattern in the string with less than O(N * k)
         * worst-case time complexity.
         * If the pattern is found, return the start index of its location. If not,
         * return False.
         */
        System.out.println("========= Q119 ==========");
        String text = "ababcababcabc";
        String patternToFind = "abcabc";

        int indexFound = stringMatch(text, patternToFind);

        if (indexFound != -1) {
            System.out.println("Pattern found at index " + indexFound);
        } else {
            System.out.println("Pattern not found");
        }

        /*
         * Q120.
         * A wall consists of several rows of bricks of various integer lengths and
         * uniform height. Your goal is to find a vertical line going from the top to
         * the bottom of the wall that cuts through the fewest number of bricks. If the
         * line goes through the edge between two bricks, this does not count as a cut.
         * For example, suppose the input is as follows, where values in each row
         * represent the lengths of bricks in that row:
         * " [[3, 5, 1, 1],    "
         * "  [2, 3, 3, 2],    "
         * "  [5, 5],          "
         * "  [4, 4, 2],       "
         * "  [1, 3, 3, 3],    "
         * "  [1, 1, 6, 1, 1]] "
         * The best we can we do here is to draw a line after the eighth brick, which
         * will only require cutting through the bricks in the third and fifth row.
         * Given an input consisting of brick lengths for each row such as the one
         * above, return the fewest number of bricks that must be cut to create a
         * vertical line.
         */
        System.out.println("========= Q120 ==========");
        List<List<Integer>> wall = new ArrayList<>();
        wall.add(Arrays.asList(3, 5, 1, 1));
        wall.add(Arrays.asList(2, 3, 3, 2));
        wall.add(Arrays.asList(5, 5));
        wall.add(Arrays.asList(4, 4, 2));
        wall.add(Arrays.asList(1, 3, 3, 3));
        wall.add(Arrays.asList(1, 1, 6, 1, 1));

        int fewestCuts = fewestBricks(wall);
        System.out.println("Fewest number of bricks to be cut: " + fewestCuts);

    }
}
