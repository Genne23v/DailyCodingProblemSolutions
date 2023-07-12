package Hard;

import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.PriorityQueue;
import java.util.Stack;
import java.util.Vector;
import java.util.Deque;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
import java.util.Comparator;

class XORNode {
    private int data;
    private long both;

    public XORNode(int data) {
        this.data = data;
    }

    public int getData() {
        return data;
    }

    public void setBoth(long both) {
        this.both = both;
    }

    public long getBoth() {
        return both;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
        left = null;
        right = null;
    }
}

public class QuestionsAndSolutions {
    // S1.
    public static int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] prefix = new int[n];
        int[] suffix = new int[n];
        int[] result = new int[n];

        prefix[0] = 1;
        for (int i = 1; i < n; i++) {
            prefix[i] = prefix[i - 1] * nums[i - 1];
        }

        suffix[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) {
            suffix[i] = suffix[i + 1] * nums[i + 1];
        }

        for (int i = 0; i < n; i++) {
            result[i] = prefix[i] * suffix[i];
        }

        return result;
    }

    // S2.
    public static int findMissingPositive(int[] nums) {
        int n = nums.length;

        // Step 1: Ignore negative numbers and numbers greater than n
        for (int i = 0; i < n; i++) {
            if (nums[i] <= 0 || nums[i] > n) {
                nums[i] = n + 1; // Mark irrelevant numbers
            }
        }

        // Step 2: Rearrange the array using index as a hash
        for (int i = 0; i < n; i++) {
            int num = Math.abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }

        // Step 3: Find the index of the first positive number (mismatch)
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }

        return n + 1;
    }

    // S3.
    static class XORLinkedList {
        private XORNode head;
        private XORNode tail;
        private Map<Integer, XORNode> nodeMap;

        public XORLinkedList() {
            nodeMap = new HashMap<>();
        }

        public void add(int element) {
            XORNode newNode = new XORNode(element);
            if (head == null) {
                head = newNode;
                tail = newNode;
            } else {
                newNode.setBoth(getPointer(tail));
                tail.setBoth(tail.getBoth() ^ getPointer(newNode));
                tail = newNode;
            }
            nodeMap.put(newNode.hashCode(), newNode);
        }

        public XORNode get(int index) {
            XORNode current = head;
            XORNode prev = null;
            for (int i = 0; i < index; i++) {
                XORNode next = dereferencePointer(current.getBoth() ^ getPointer(prev));
                prev = current;
                current = next;
            }
            return current;
        }

        private long getPointer(XORNode node) {
            return node != null ? node.hashCode() : 0;
        }

        private XORNode dereferencePointer(long pointer) {
            if (pointer == 0) {
                return null;
            }
            return nodeMap.get((int) pointer);
        }
    }

    // S4.
    public static int largestSumNonAdjacent(int[] nums) {
        int inclusive = 0; // sum including the current element
        int exclusive = 0; // sum excluding the current element

        for (int num : nums) {
            int temp = inclusive;
            inclusive = Math.max(inclusive, exclusive + num);
            exclusive = temp;
        }

        return Math.max(inclusive, exclusive);
    }

    // S5.
    public static int countWays(int n, int[] X) {
        int[] dp = new int[n + 1];
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < X.length; j++) {
                if (i >= X[j]) {
                    dp[i] += dp[i - X[j]];
                }
            }
        }

        return dp[n];
    }

    // S6.
    public static int longestSubstringWithKDistinct(String s, int k) {
        if (s == null || s.length() == 0 || k <= 0) {
            return 0;
        }

        int maxLength = 0;
        int start = 0;
        int distinctCount = 0;
        Map<Character, Integer> charCount = new HashMap<>();

        for (int end = 0; end < s.length(); end++) {
            char c = s.charAt(end);
            charCount.put(c, charCount.getOrDefault(c, 0) + 1);

            if (charCount.get(c) == 1) {
                distinctCount++;
            }

            while (distinctCount > k) {
                char startChar = s.charAt(start);
                charCount.put(startChar, charCount.get(startChar) - 1);

                if (charCount.get(startChar) == 0) {
                    distinctCount--;
                }

                start++;
            }

            maxLength = Math.max(maxLength, end - start + 1);
        }

        return maxLength;
    }

    // S7.
    public static int lengthLongestPath(String input) {
        if (input == null || input.length() == 0) {
            return 0;
        }

        String[] paths = input.split("\n");
        Deque<Integer> stack = new ArrayDeque<>();
        int maxLength = 0;

        for (String path : paths) {
            int level = getLevel(path);
            while (stack.size() > level) {
                stack.pop();
            }

            int currLength = (stack.isEmpty() ? 0 : stack.peek()) + path.length() - level + 1;
            stack.push(currLength);

            if (isFile(path)) {
                maxLength = Math.max(maxLength, currLength - 1);
            }
        }

        return maxLength;
    }

    private static int getLevel(String path) {
        int level = 0;
        int i = 0;
        while (i < path.length() && path.charAt(i) == '\t') {
            level++;
            i++;
        }
        return level;
    }

    private static boolean isFile(String path) {
        return path.contains(".");
    }

    // S8.
    public static void printMaxOfSubarrays(int[] nums, int k) {
        Deque<Integer> deque = new LinkedList<>();
        int n = nums.length;

        for (int i = 0; i < k; i++) {
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.removeLast();
            }
            deque.addLast(i);
        }

        for (int i = k; i < n; i++) {
            System.out.print(nums[deque.peekFirst()] + " ");

            // Remove elements outside the current window
            while (!deque.isEmpty() && deque.peekFirst() <= i - k) {
                deque.removeFirst();
            }

            // Remove smaller elements from the deque
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.removeLast();
            }

            deque.addLast(i);
        }

        // Print the maximum of the last subarray
        System.out.print(nums[deque.peekFirst()]);
    }

    // S9.
    public static boolean isMatch(String text, String pattern) {
        if (pattern.isEmpty()) {
            return text.isEmpty();
        }

        boolean firstMatch = !text.isEmpty() &&
                (pattern.charAt(0) == text.charAt(0) || pattern.charAt(0) == '.');

        if (pattern.length() >= 2 && pattern.charAt(1) == '*') {
            return (isMatch(text, pattern.substring(2)) ||
                    (firstMatch && isMatch(text.substring(1), pattern)));
        } else {
            return firstMatch && isMatch(text.substring(1), pattern.substring(1));
        }
    }

    // S10.
    public static boolean hasArbitrage(double[][] rates) {
        int n = rates.length;
        double[] dist = new double[n];
        Arrays.fill(dist, Double.MAX_VALUE);
        dist[0] = 0;

        // Relax edges repeatedly (Bellman-Ford algorithm)
        for (int i = 0; i < n - 1; i++) {
            for (int u = 0; u < n; u++) {
                for (int v = 0; v < n; v++) {
                    if (dist[u] != Double.MAX_VALUE && rates[u][v] != 0) {
                        double newDist = dist[u] + Math.log(rates[u][v]);
                        if (newDist < dist[v]) {
                            dist[v] = newDist;
                        }
                    }
                }
            }
        }

        // Check for negative cycles
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (dist[u] != Double.MAX_VALUE && rates[u][v] != 0) {
                    double newDist = dist[u] + Math.log(rates[u][v]);
                    if (newDist < dist[v]) {
                        return true; // Negative cycle exists
                    }
                }
            }
        }

        return false;
    }

    // S11.
    public static void segregateColors(char[] colors) {
        int low = 0;
        int mid = 0;
        int high = colors.length - 1;

        while (mid <= high) {
            if (colors[mid] == 'R') {
                swap(colors, low, mid);
                low++;
                mid++;
            } else if (colors[mid] == 'G') {
                mid++;
            } else if (colors[mid] == 'B') {
                swap(colors, mid, high);
                high--;
            }
        }
    }

    public static void swap(char[] colors, int i, int j) {
        char temp = colors[i];
        colors[i] = colors[j];
        colors[j] = temp;
    }

    // S12.
    public static int countNQueens(int n) {
        int[] queens = new int[n];
        return backtrack(queens, 0);
    }

    public static int backtrack(int[] queens, int row) {
        int count = 0;
        if (row == queens.length) {
            return 1;
        }

        for (int col = 0; col < queens.length; col++) {
            if (isSafe(queens, row, col)) {
                queens[row] = col;
                count += backtrack(queens, row + 1);
            }
        }

        return count;
    }

    public static boolean isSafe(int[] queens, int row, int col) {
        for (int i = 0; i < row; i++) {
            if (queens[i] == col || queens[i] - i == col - row || queens[i] + i == col + row) {
                return false;
            }
        }
        return true;
    }

    // S13.
    public static int findNonDuplicated(int[] nums) {
        int ones = 0; // Count only bits that appear once
        int twos = 0; // Count only bits that appear twice

        for (int num : nums) {
            ones = (ones ^ num) & ~twos;
            twos = (twos ^ num) & ~ones;
        }

        return ones;
    }

    // S14.
    public static List<Integer> findSubset(int[] nums, int target) {
        List<Integer> subset = new ArrayList<>();
        boolean targetFound = backtrack(nums, target, 0, subset);
        return targetFound ? subset : null;
    }

    private static boolean backtrack(int[] nums, int target, int index, List<Integer> subset) {
        if (target == 0) {
            return true;
        }

        for (int i = index; i < nums.length; i++) {
            if (nums[i] <= target) {
                subset.add(nums[i]);
                boolean targetFound = backtrack(nums, target - nums[i], i + 1, subset);

                if (targetFound) {
                    return true;
                }
                subset.remove(subset.size() - 1);
            }
        }
        return false;
    }

    // S15.
    public static String longestPalindromicSubstring(String s) {
        if (s == null || s.length() < 2) {
            return s;
        }

        int start = 0;
        int maxLength = 1;
        int n = s.length();
        boolean[][] dp = new boolean[n][n];

        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }

        for (int i = 0; i < n - 1; i++) {
            if (s.charAt(i) == s.charAt(i + 1)) {
                dp[i][i + 1] = true;
                start = i;
                maxLength = 2;
            }
        }

        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]) {
                    dp[i][j] = true;

                    if (len > maxLength) {
                        start = i;
                        maxLength = len;
                    }
                }
            }
        }

        return s.substring(start, start + maxLength);
    }

    // S16.
    static class LRUCache {
        class Node {
            int key;
            int value;
            Node prev;
            Node next;

            public Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }

        private Map<Integer, Node> cache;
        private int capacity;
        private Node head;
        private Node tail;

        public LRUCache(int capacity) {
            this.cache = new HashMap<>();
            this.capacity = capacity;

            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.next = tail;
            tail.prev = head;
        }

        public int get(int key) {
            if (cache.containsKey(key)) {
                Node node = cache.get(key);
                removeNode(node);
                addNodeToHead(node);
                return node.value;
            }
            return -1;
        }

        public void set(int key, int value) {
            if (cache.containsKey(key)) {
                Node node = cache.get(key);
                node.value = value;
                removeNode(node);
                addNodeToHead(node);
            } else {
                Node newNode = new Node(key, value);
                if (cache.size() >= capacity) {
                    Node tailNode = tail.prev;
                    removeNode(tailNode);
                    cache.remove(tail.prev.key);
                }
                cache.put(key, newNode);
                addNodeToHead(newNode);
            }
        }

        private void removeNode(Node node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        private void addNodeToHead(Node node) {
            node.next = head.next;
            node.prev = head;
            head.next.prev = node;
            head.next = node;
        }
    }

    // S17.
    public static boolean solveSudoku(int[][] grid) {
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                if (grid[row][col] == 0) {
                    for (int digit = 1; digit <= 9; digit++) {
                        if (isValid(grid, row, col, digit)) {
                            grid[row][col] = digit;
                            if (solveSudoku(grid)) {
                                return true;
                            }
                            grid[row][col] = 0;
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    public static boolean isValid(int[][] grid, int row, int col, int digit) {
        for (int i = 0; i < 9; i++) {
            if (grid[row][i] == digit || grid[i][col] == digit) {
                return false;
            }
        }

        int subgridStartRow = 3 * (row / 3);
        int subgridStartCol = 3 * (col / 3);

        for (int i = subgridStartRow; i < subgridStartRow + 3; i++) {
            for (int j = subgridStartCol; j < subgridStartCol + 3; j++) {
                if (grid[i][j] == digit) {
                    return false;
                }
            }
        }

        return true;
    }

    public static void printGrid(int[][] grid) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                System.out.print(grid[i][j] + " ");
            }
            System.out.println();
        }
    }

    // S18.
    // Solution in Q18

    // S19.
    static class KnightsTour {
        private static final int[] ROW_MOVES = { 2, 1, -1, -2, -2, -1, 1, 2 };
        private static final int[] COL_MOVES = { 1, 2, 2, 1, -1, -2, -2, -1 };

        public int countKnightTours(int n) {
            int[][] board = new int[n][n];
            int count = 0;

            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) {
                    count += findTours(board, row, col, 1);
                }
            }

            return count;
        }

        private int findTours(int[][] board, int row, int col, int moveCount) {
            int n = board.length;

            if (moveCount == n * n) {
                return 1;
            }

            int count = 0;
            board[row][col] = moveCount;

            for (int i = 0; i < 8; i++) {
                int nextRow = row + ROW_MOVES[i];
                int nextCol = col + COL_MOVES[i];

                if (isValidMove(board, nextRow, nextCol)) {
                    count += findTours(board, nextRow, nextCol, moveCount + 1);
                }
            }

            board[row][col] = 0;
            return count;
        }

        private boolean isValidMove(int[][] board, int row, int col) {
            int n = board.length;
            return (row >= 0 && row < n && col >= 0 && col < n && board[row][col] == 0);
        }
    }

    // S20.
    static class LFUCache {
        private int capacity;
        private int minFrequency;
        private Map<Integer, Integer> keyToValue;
        private Map<Integer, Integer> keyToFrequency;
        private Map<Integer, LinkedHashSet<Integer>> frequencyToKeys;

        public LFUCache(int capacity) {
            this.capacity = capacity;
            this.minFrequency = 0;
            this.keyToValue = new HashMap<>();
            this.keyToFrequency = new HashMap<>();
            this.frequencyToKeys = new HashMap<>();
        }

        public int get(int key) {
            if (!keyToValue.containsKey(key)) {
                return -1;
            }

            int frequency = keyToFrequency.get(key);
            keyToFrequency.put(key, frequency + 1);
            frequencyToKeys.get(frequency).remove(key);

            if (frequencyToKeys.get(frequency).isEmpty()) {
                frequencyToKeys.remove(frequency);

                if (minFrequency == frequency) {
                    minFrequency++;
                }
            }

            frequencyToKeys.putIfAbsent(frequency + 1, new LinkedHashSet<>());
            frequencyToKeys.get(frequency + 1).add(key);

            return keyToValue.get(key);
        }

        public void set(int key, int value) {
            if (capacity <= 0) {
                return;
            }

            if (keyToValue.containsKey(key)) {
                keyToValue.put(key, value);
                get(key);
                return;
            }

            if (keyToValue.size() >= capacity) {
                int evictKey = frequencyToKeys.get(minFrequency).iterator().next();
                frequencyToKeys.get(minFrequency).remove(evictKey);
                keyToValue.remove(evictKey);
                keyToFrequency.remove(evictKey);
            }

            keyToValue.put(key, value);
            keyToFrequency.put(key, 1);
            frequencyToKeys.putIfAbsent(1, new LinkedHashSet<>());
            frequencyToKeys.get(1).add(key);
            minFrequency = 1;
        }
    }

    // S21.
    static class DirectedGraph {
        private Map<Integer, Set<Integer>> adjacencyList;
        private int[] nodeValues;

        public DirectedGraph(String nodes, int[][] edges) {
            adjacencyList = new HashMap<>();
            nodeValues = new int[nodes.length()];

            for (int i = 0; i < nodes.length(); i++) {
                adjacencyList.put(i, new HashSet<>());
                nodeValues[i] = nodes.charAt(i) - '0';
            }

            for (int[] edge : edges) {
                adjacencyList.get(edge[0]).add(edge[1]);
            }
        }

        public Integer findLargestValuePath() {
            int numNodes = nodeValues.length;
            int[] visited = new int[numNodes];
            Map<Character, Integer> freqMap = new HashMap<>();
            int maxValue = 0;

            for (int i = 0; i < numNodes; i++) {
                if (visited[i] == 0) {
                    Integer pathValue = dfs(i, visited, freqMap);

                    if (pathValue == null) {
                        return null;
                    }
                    maxValue = Math.max(maxValue, pathValue);
                }
            }

            return maxValue;
        }

        public Integer dfs(int node, int[] visited, Map<Character, Integer> freqMap) {
            visited[node] = 1;
            char currNode = (char) nodeValues[node];
            freqMap.put(currNode, freqMap.getOrDefault(currNode, 0) + 1);
            Set<Integer> neighbors = adjacencyList.getOrDefault(node, Collections.emptySet());
            Integer maxValue = freqMap.get(currNode);

            for (int neighbor : neighbors) {
                if (visited[neighbor] == 1) {
                    return null;
                }

                Integer pathValue = dfs(neighbor, visited, freqMap);
                if (pathValue == null) {
                    return null;
                }
                maxValue = Math.max(maxValue, pathValue);
            }

            visited[node] = 0;
            freqMap.put(currNode, freqMap.get(currNode) - 1);
            if (freqMap.get(currNode) == 0) {
                freqMap.remove(currNode);
            }

            return maxValue;
        }
    }

    // S22.
    public static int lengthOfLIS(int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }

        int[] dp = new int[n];
        dp[0] = 1;
        int maxLength = 1;

        for (int i = 1; i < n; i++) {
            int maxVal = 0;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    maxVal = Math.max(maxVal, dp[j]);
                }
            }
            dp[i] = maxVal + 1;
            maxLength = Math.max(maxLength, dp[i]);
        }

        return maxLength;
    }

    // S23.
    public static boolean validateRules(List<String> rules) {
        Map<String, Set<String>> relationships = new HashMap<>();

        for (String rule : rules) {
            String[] parts = rule.split(" ");
            String point1 = parts[0];
            String direction = parts[1];
            String point2 = parts[2];

            if (!validateRule(relationships, point1, direction, point2)) {
                return false;
            }
        }

        return true;
    }

    private static boolean validateRule(Map<String, Set<String>> relationships, String point1, String direction,
            String point2) {
        Set<String> existingDirections1 = relationships.getOrDefault(point1, new HashSet<>());
        Set<String> existingDirections2 = relationships.getOrDefault(point2, new HashSet<>());

        if (existingDirections1.contains(directionOpposite(direction)) || existingDirections2.contains(direction)) {
            return false;
        }

        existingDirections1.add(direction);
        existingDirections2.add(directionOpposite(direction));

        relationships.put(point1, existingDirections1);
        relationships.put(point2, existingDirections2);

        return true;
    }

    private static String directionOpposite(String direction) {
        switch (direction) {
            case "N":
                return "S";
            case "S":
                return "N";
            case "E":
                return "W";
            case "W":
                return "E";
            case "NE":
                return "SW";
            case "SW":
                return "NE";
            case "NW":
                return "SE";
            case "SE":
                return "NW";
            default:
                return "";
        }
    }

    // S24.
    public static List<String> findCourseOrder(Map<String, List<String>> prerequisites) {
        Map<String, List<String>> graph = buildGraph(prerequisites);
        Set<String> visited = new HashSet<>();
        Set<String> visiting = new HashSet<>();
        List<String> courseOrder = new ArrayList<>();

        for (String course : graph.keySet()) {
            if (!visited.contains(course) && !dfs(course, graph, visited, visiting, courseOrder)) {
                return null;
            }
        }

        Collections.reverse(courseOrder);
        return courseOrder;
    }

    private static Map<String, List<String>> buildGraph(Map<String, List<String>> prerequisites) {
        Map<String, List<String>> graph = new HashMap<>();

        for (String course : prerequisites.keySet()) {
            graph.put(course, new ArrayList<>());
        }

        for (String course : prerequisites.keySet()) {
            for (String prerequisite : prerequisites.get(course)) {
                graph.get(prerequisite).add(course);
            }
        }

        return graph;
    }

    private static boolean dfs(String course, Map<String, List<String>> graph, Set<String> visited,
            Set<String> visiting, List<String> courseOrder) {
        visiting.add(course);

        for (String prerequisite : graph.get(course)) {

            if (visiting.contains(prerequisite)) {
                return false;
            }

            if (!visited.contains(prerequisite) && !dfs(prerequisite, graph, visited, visiting, courseOrder)) {
                return false;
            }
        }

        visiting.remove(course);
        visited.add(course);
        courseOrder.add(course);
        return true;
    }

    // S25.
    static class Result {
        int size;
        int max;
        int min;
        boolean isBST;

        Result(int size, int max, int min, boolean isBST) {
            this.size = size;
            this.max = max;
            this.min = min;
            this.isBST = isBST;
        }
    }

    public static int largestBST(TreeNode root) {
        return largestBSTHelper(root).size;
    }

    public static Result largestBSTHelper(TreeNode root) {
        if (root == null) {
            return new Result(0, Integer.MIN_VALUE, Integer.MAX_VALUE, true);
        }

        Result left = largestBSTHelper(root.left);
        Result right = largestBSTHelper(root.right);

        if (!left.isBST || !right.isBST || root.val < left.max || root.val > right.min) {
            return new Result(Math.max(left.size, right.size), 0, 0, false);
        }

        int size = left.size + right.size + 1;
        int min = root.left != null ? left.min : root.val;
        int max = root.right != null ? right.max : root.val;

        return new Result(size, max, min, true);
    }

    // S26.
    public static void nextPermutation(int[] nums) {
        int n = nums.length;
        int i = n - 2;

        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }

        if (i >= 0) {
            int j = n - 1;
            while (j > i && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }

        reverse(nums, i + 1, n - 1);
    }

    private static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private static void reverse(int[] nums, int start, int end) {
        while (start < end) {
            swap(nums, start, end);
            start++;
            end--;
        }
    }

    // S27.
    public static List<Integer> findAnagramIndices(String s, String w) {
        List<Integer> result = new ArrayList<>();

        if (s.length() == 0 || w.length() == 0 || s.length() < w.length()) {
            return result;
        }

        Map<Character, Integer> targetFreqMap = new HashMap<>();
        Map<Character, Integer> windowFreqMap = new HashMap<>();

        for (char ch : w.toCharArray()) {
            targetFreqMap.put(ch, targetFreqMap.getOrDefault(ch, 0) + 1);
        }

        int windowSize = w.length();

        for (int i = 0; i < windowSize; i++) {
            char ch = s.charAt(i);
            windowFreqMap.put(ch, windowFreqMap.getOrDefault(ch, 0) + 1);
        }

        if (isAnagram(targetFreqMap, windowFreqMap)) {
            result.add(0);
        }

        for (int i = windowSize; i < s.length(); i++) {
            char incoming = s.charAt(i);
            char outgoing = s.charAt(i - windowSize);

            windowFreqMap.put(incoming, windowFreqMap.getOrDefault(incoming, 0) + 1);
            windowFreqMap.put(outgoing, windowFreqMap.get(outgoing) - 1);

            if (windowFreqMap.get(outgoing) == 0) {
                windowFreqMap.remove(outgoing);
            }

            if (isAnagram(targetFreqMap, windowFreqMap)) {
                result.add(i - windowSize + 1);
            }
        }

        return result;
    }

    private static boolean isAnagram(Map<Character, Integer> targetFreqMap, Map<Character, Integer> windowFreqMap) {
        return targetFreqMap.equals(windowFreqMap);
    }

    // S28.
    static class TreeNodeWithParent {
        int val;
        TreeNodeWithParent left;
        TreeNodeWithParent right;
        TreeNodeWithParent parent;

        TreeNodeWithParent(int val) {
            this.val = val;
            this.left = null;
            this.right = null;
            this.parent = null;
        }
    }

    public static TreeNodeWithParent findLowestCommonAncestor(TreeNodeWithParent root, TreeNodeWithParent p,
            TreeNodeWithParent q) {
        if (root == null || p == null || q == null) {
            return null;
        }

        List<TreeNodeWithParent> pathToP = getPathToRoot(p);
        List<TreeNodeWithParent> pathToQ = getPathToRoot(q);

        int i = 0;
        while (i < pathToP.size() && i < pathToQ.size()) {
            if (pathToP.get(i) != pathToQ.get(i)) {
                break;
            }
            i++;
        }

        if (i > 0) {
            return pathToP.get(i - 1);
        }

        return null;
    }

    private static List<TreeNodeWithParent> getPathToRoot(TreeNodeWithParent node) {
        List<TreeNodeWithParent> path = new ArrayList<>();

        while (node != null) {
            path.add(node);
            node = node.parent;
        }

        Collections.reverse(path);

        return path;
    }

    // S29.
    public static String reverseWordsWithDelimiters(String str, Set<Character> delimiters) {
        String[] words = str.split("[\\" + getDelimitersAsString(delimiters) + "]+");
        String[] separators = str.split("[^" + getDelimitersAsString(delimiters) + "]+");
        separators = Arrays.stream(separators).filter(s -> !s.isEmpty()).toArray(String[]::new);

        reverseArray(words);
        
        StringBuilder reversed = new StringBuilder();
        int i = 0, j = 0;

        while (i < words.length || j < separators.length) {
            if (i < words.length) {
                reversed.append(words[i++]);
            }
            if (j < separators.length) {
                reversed.append(separators[j++]);
            }
        }

        return reversed.toString();
    }

    private static String getDelimitersAsString(Set<Character> delimiters) {
        StringBuilder sb = new StringBuilder();
        for (char delimiter : delimiters) {
            sb.append(delimiter);
        }
        return sb.toString();
    }

    private static void reverseArray(String[] arr) {
        int start = 0;
        int end = arr.length - 1;

        while (start < end) {
            String temp = arr[start];
            arr[start] = arr[end];
            arr[end] = temp;
            start++;
            end--;
        }
    }

    // S30.
    public static boolean isSubtree(TreeNode s, TreeNode t) {
        if (s == null) {
            return false;
        }

        if (isSameTree(s, t)) {
            return true;
        }

        return isSubtree(s.left, t) || isSubtree(s.right, t);
    }

    private static boolean isSameTree(TreeNode s, TreeNode t) {
        if (s == null && t == null) {
            return true;
        }

        if (s == null || t == null) {
            return false;
        }

        return s.val == t.val && isSameTree(s.left, t.left) && isSameTree(s.right, t.right);
    }

    // S31.
    public static boolean canMakePalindrome(String s, int k) {
        int n = s.length();
        int[][] dp = new int[n][n];

        for (int i = 0; i < n; i++) {
            dp[i][i] = 0;
        }

        for (int len = 2; len <= n; len++) {
            for (int i = 0; i < n - len + 1; i++) {
                int j = i + len - 1;

                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
                }
            }
        }

        return dp[0][n - 1] <= k;
    }

    // S32.
    public static boolean isNumber(String s) {
        s = s.trim();
        String pattern = "[-+]?(\\d+\\.?|\\.\\d+)\\d*(e[-+]?\\d+)?";
        return s.matches(pattern);
    }

    // S33.
    public static int coinChange(int n) {
        int[] coins = { 25, 10, 5, 1 };

        int[] dp = new int[n + 1];
        Arrays.fill(dp, n + 1);

        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
                }
            }
        }

        return dp[n] > n ? -1 : dp[n];
    }

    // S34.
    static class MultiStack {
        private int[] list;
        private int[] tops;
        private int stackSize;
        private int numStacks;

        public MultiStack(int stackSize, int numStacks) {
            this.stackSize = stackSize;
            this.numStacks = numStacks;
            this.list = new int[stackSize * numStacks];
            this.tops = new int[numStacks];
            Arrays.fill(tops, -1);
        }

        public void push(int item, int stackNumber) {
            if (isFull(stackNumber)) {
                System.out.println("Stack " + stackNumber + " is full. Cannot push item: " + item);
                return;
            }

            int index = getTopIndex(stackNumber);
            index++;
            list[stackSize * stackNumber + index] = item;
            tops[stackNumber] = index;
        }

        public int pop(int stackNumber) {
            if (isEmpty(stackNumber)) {
                System.out.println("Stack " + stackNumber + " is empty. Cannot pop item.");
                return -1;
            }

            int index = getTopIndex(stackNumber);
            int item = list[stackSize * stackNumber + index];
            tops[stackNumber] = index - 1;
            return item;
        }

        public boolean isEmpty(int stackNumber) {
            return tops[stackNumber] == -1;
        }

        public boolean isFull(int stackNumber) {
            return tops[stackNumber] == (stackSize * (stackNumber + 1)) - 1;
        }

        private int getTopIndex(int stackNumber) {
            return tops[stackNumber];
        }
    }

    // S35.
    public static boolean isBalanced(String str) {
        int minOpen = 0;
        int maxOpen = 0;

        for (char c : str.toCharArray()) {
            if (c == '(') {
                minOpen++;
                maxOpen++;
            } else if (c == ')') {
                minOpen = Math.max(minOpen - 1, 0);
                maxOpen--;
            } else if (c == '*') {
                minOpen = Math.max(minOpen - 1, 0);
                maxOpen++;
            }

            if (maxOpen < 0) {
                return false; // More closing parentheses encountered than open parentheses
            }
        }

        return minOpen == 0;
    }

    // S36.
    public static void sort(List<Integer> lst) {
        int n = lst.size();
        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (lst.get(j) < lst.get(minIndex)) {
                    minIndex = j;
                }
            }
            reverse(lst, i, minIndex);
        }
    }

    public static void reverse(List<Integer> lst, int start, int end) {
        while (start < end) {
            int temp = lst.get(start);
            lst.set(start, lst.get(end));
            lst.set(end, temp);
            start++;
            end--;
        }
    }

    // S37.
    static class SublistSum {
        private List<Integer> prefixSums;

        public SublistSum(List<Integer> lst) {
            prefixSums = new ArrayList<>(lst.size() + 1);
            prefixSums.add(0);
            int sum = 0;
            for (int num : lst) {
                sum += num;
                prefixSums.add(sum);
            }
        }

        public int sum(int i, int j) {
            if (i < 0 || j > prefixSums.size() || i > j) {
                throw new IllegalArgumentException("Invalid sublist range");
            }
            return prefixSums.get(j) - prefixSums.get(i);
        }
    }

    // S38.
    static class Point {
        int x;
        int y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public static List<Point> findNearestPoints(List<Point> points, Point centralPoint, int k) {
        PriorityQueue<Point> pq = new PriorityQueue<>(k,
                Comparator.comparingDouble(p -> -calculateDistance(p, centralPoint)));

        for (Point point : points) {
            pq.offer(point);
            if (pq.size() > k) {
                pq.poll();
            }
        }

        return new ArrayList<>(pq);
    }

    private static double calculateDistance(Point p1, Point p2) {
        int dx = p1.x - p2.x;
        int dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    // S39.
    public static int findSmallestDistance(String text, String word1, String word2) {
        String[] words = text.split("\\s+");
        int minDistance = Integer.MAX_VALUE;
        int prevIndex = -1;

        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                if (prevIndex != -1 && i - prevIndex < minDistance) {
                    minDistance = i - prevIndex;
                }
                prevIndex = i;
            } else if (words[i].equals(word2)) {
                if (prevIndex != -1 && i - prevIndex < minDistance) {
                    minDistance = i - prevIndex;
                }
                prevIndex = i;
            }
        }

        return minDistance - 1;
    }

    // S40.
    private static class GraphNode {
        char id;
        List<TreeNodeEdge> edges;

        public GraphNode(char id) {
            this.id = id;
            edges = new ArrayList<>();
        }
    }

    private static class TreeNodeEdge {
        GraphNode destination;
        int weight;

        public TreeNodeEdge(GraphNode destination, int weight) {
            this.destination = destination;
            this.weight = weight;
        }
    }

    private static int longestPath = 0;

    public static int calculateLongestPath(GraphNode root) {
        if (root == null) {
            return 0;
        }

        calculatePath(root, null);

        return longestPath;
    }

    private static int calculatePath(GraphNode node, GraphNode parent) {
        if (node.edges.size() == 1 && node != parent) {
            return 0;
        }

        int maxPath1 = 0;
        int maxPath2 = 0;

        for (TreeNodeEdge edge : node.edges) {
            if (edge.destination != parent) {
                int subPath = calculatePath(edge.destination, node) + edge.weight;
                if (subPath > maxPath1) {
                    maxPath2 = maxPath1;
                    maxPath1 = subPath;
                } else if (subPath > maxPath2) {
                    maxPath2 = subPath;
                }
            }
        }

        longestPath = Math.max(longestPath, maxPath1 + maxPath2);

        return maxPath1;
    }

    // S41.
    public static int evaluateRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();

        for (String token : tokens) {
            if (isOperator(token)) {
                int operand2 = stack.pop();
                int operand1 = stack.pop();
                int result = performOperation(operand1, operand2, token);
                stack.push(result);
            } else {
                int operand = Integer.parseInt(token);
                stack.push(operand);
            }
        }

        return stack.pop();
    }

    private static boolean isOperator(String token) {
        return token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/");
    }

    private static int performOperation(int operand1, int operand2, String operator) {
        switch (operator) {
            case "+":
                return operand1 + operand2;
            case "-":
                return operand1 - operand2;
            case "*":
                return operand1 * operand2;
            case "/":
                return operand1 / operand2;
        }
        return 0;
    }

    // S42.
    public static List<List<Integer>> findPalindromePairs(String[] words) {
        List<List<Integer>> pairs = new ArrayList<>();

        for (int i = 0; i < words.length; i++) {
            for (int j = 0; j < words.length; j++) {
                if (i != j && isPalindrome(words[i] + words[j])) {
                    List<Integer> pair = new ArrayList<>();
                    pair.add(i);
                    pair.add(j);
                    pairs.add(pair);
                }
            }
        }

        return pairs;
    }

    private static boolean isPalindrome(String word) {
        int i = 0;
        int j = word.length() - 1;

        while (i < j) {
            if (word.charAt(i) != word.charAt(j)) {
                return false;
            }
            i++;
            j--;
        }

        return true;
    }

    // S43.
    private static double calculateExpectedValue(int target1, int target2, int numSimulations) {
        int totalRolls = 0;

        for (int i = 0; i < numSimulations; i++) {
            int rolls = simulateGame(target1, target2);
            totalRolls += rolls;
        }

        return (double) totalRolls / numSimulations;
    }

    private static int simulateGame(int target1, int target2) {
        Random random = new Random();
        int rolls = 0;
        boolean target1Found = false;

        while (true) {
            int roll = random.nextInt(6) + 1;
            rolls++;

            if (target1Found && roll == target2) {
                break;
            }

            target1Found = (roll == target1);
        }

        return rolls;
    }

    // S44.
    public static List<String> splitIntoPalindromes(String s) {
        List<String> result = new ArrayList<>();
        splitIntoPalindromesHelper(s, 0, new ArrayList<>(), result);
        return result;
    }

    private static void splitIntoPalindromesHelper(String s, int start, List<String> current, List<String> result) {
        if (start == s.length()) {
            result.clear();
            result.addAll(current);
            return;
        }

        for (int i = start + 1; i <= s.length(); i++) {
            String substring = s.substring(start, i);
            if (isPalindrome(substring)) {
                current.add(substring);
                splitIntoPalindromesHelper(s, i, current, result);
                current.remove(current.size() - 1);
            }
        }
    }

    // S45.
    // Solution in Q45

    // S46.
    public static int minSubsetSumDifference(int[] nums) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }

        boolean[][] dp = new boolean[nums.length + 1][totalSum + 1];

        for (int i = 0; i <= nums.length; i++) {
            dp[i][0] = true;
        }

        for (int i = 1; i <= nums.length; i++) {
            for (int j = 1; j <= totalSum; j++) {
                if (nums[i - 1] <= j) {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        int minDiff = Integer.MAX_VALUE;

        for (int j = totalSum / 2; j >= 0; j--) {
            if (dp[nums.length][j]) {
                minDiff = totalSum - 2 * j;
                break;
            }
        }

        printSubsets(nums, dp, (totalSum - minDiff) / 2);

        return minDiff;
    }

    public static void printSubsets(int[] nums, boolean[][] dp, int median) {
        List<Integer> subset1 = new ArrayList<>();
        List<Integer> subset2 = new ArrayList<>();
        int i = nums.length;
        int j = median;

        while (i > 0 && j > 0) {
            if (nums[i - 1] <= j && dp[i - 1][j - nums[i - 1]]) {
                subset1.add(nums[i - 1]);
                j -= nums[i - 1];
            }
            i--;
        }

        subset2.addAll(Arrays.asList(Arrays.stream(nums).boxed().toArray(Integer[]::new)));
        subset2.removeAll(subset1);

        System.out.println("Subset 1: " + subset1);
        System.out.println("Subset 2: " + subset2);
    }

    // S47.
    public static int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        int[][] dp = new int[n][2];

        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }

        return dp[n - 1][0];
    }

    // S48.
    public static int countElements(int[][] matrix, int i1, int j1, int i2, int j2) {
        int count = 0;
        int num1 = matrix[i1][j1];
        int num2 = matrix[i2][j2];

        for (int[] row : matrix) {
            for (int num : row) {
                if (num < num1 || num > num2) {
                    count++;
                }
            }
        }

        return count;
    }

    // S49.
    public static String balanceParentheses(String s) {
        StringBuilder balancedString = new StringBuilder();
        Stack<Character> stack = new Stack<>();

        for (char c : s.toCharArray()) {
            if (c == '(') {
                stack.push(c);
                balancedString.append(c);
            } else if (c == ')') {
                if (!stack.isEmpty() && stack.peek() == '(') {
                    stack.pop();
                    balancedString.append(c);
                } else {
                    balancedString.append('(').append(c);
                }
            } else {
                balancedString.append(c);
            }
        }

        while (!stack.isEmpty()) {
            balancedString.append(')');
            stack.pop();
        }

        return balancedString.toString();
    }

    // S50.
    static class Interval {
        int start;
        int end;

        Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    public static List<Integer> findStabPoints(List<Interval> intervals) {
        intervals.sort(Comparator.comparingInt(a -> a.end));
        List<Integer> points = new ArrayList<>();
        int currentPoint = intervals.get(0).end;

        for (Interval interval : intervals) {
            if (interval.start > currentPoint) {
                points.add(currentPoint);
                currentPoint = interval.end;
            }
        }

        points.add(currentPoint);
        return points;
    }

    // S51.
    public static int longestCommonSubsequenceLength(String text1, String text2, String text3) {
        int m = text1.length();
        int n = text2.length();
        int p = text3.length();

        int[][][] dp = new int[m + 1][n + 1][p + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                for (int k = 1; k <= p; k++) {
                    if (text1.charAt(i - 1) == text2.charAt(j - 1) && text1.charAt(i - 1) == text3.charAt(k - 1)) {
                        dp[i][j][k] = dp[i - 1][j - 1][k - 1] + 1;
                    } else {
                        dp[i][j][k] = Math.max(Math.max(dp[i - 1][j][k], dp[i][j - 1][k]), dp[i][j][k - 1]);
                    }
                }
            }
        }

        return dp[m][n][p];
    }

    // S52.
    public static int findSparseNumber(int N) {
        String binary = Integer.toBinaryString(N);
        char[] binaryArray = binary.toCharArray();

        for (int i = 1; i < binaryArray.length; i++) {
            if (binaryArray[i] == '1' && binaryArray[i - 1] == '1') {
                for (int j = i; j < binaryArray.length; j = j + 2) {
                    binaryArray[j] = '0';
                    if (j + 1 < binaryArray.length) {
                        binaryArray[j + 1] = '1';
                    }
                }
                return Integer.parseInt(String.valueOf(binaryArray), 2);
            }
        }
        return N;
    }

    // S53.
    static class Connect4 {
        private static final int ROWS = 6;
        private static final int COLUMNS = 7;

        private static final char EMPTY = '-';
        private static final char RED = 'R';
        private static final char BLACK = 'B';

        private static char[][] board;
        private static char currentPlayer;

        private static void initializeGame() {
            board = new char[ROWS][COLUMNS];
            for (char[] row : board) {
                java.util.Arrays.fill(row, EMPTY);
            }
            currentPlayer = RED;
        }

        private static void playGame() {
            boolean gameEnded = false;
            Scanner scanner = new Scanner(System.in);

            while (!gameEnded) {
                displayBoard();
                System.out.println("Player " + currentPlayer + "'s turn");
                int column = getPlayerMove(scanner);

                if (columnIsValid(column) && columnIsNotFull(column)) {
                    int row = dropDisc(column);
                    if (checkWin(row, column)) {
                        displayBoard();
                        System.out.println("Player " + currentPlayer + " wins!");
                        gameEnded = true;
                    } else if (boardIsFull()) {
                        displayBoard();
                        System.out.println("It's a draw!");
                        gameEnded = true;
                    } else {
                        switchPlayer();
                    }
                } else {
                    System.out.println("Invalid move. Please try again.");
                }
            }

            scanner.close();
        }

        private static void displayBoard() {
            for (int i = ROWS - 1; i >= 0; i--) {
                for (int j = 0; j < COLUMNS; j++) {
                    System.out.print(board[i][j] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }

        private static int getPlayerMove(Scanner scanner) {
            System.out.print("Enter a column (0-6): ");
            return scanner.nextInt();
        }

        private static boolean columnIsValid(int column) {
            return column >= 0 && column < COLUMNS;
        }

        private static boolean columnIsNotFull(int column) {
            return board[ROWS - 1][column] == EMPTY;
        }

        private static int dropDisc(int column) {
            int row = 0;
            while (row < ROWS) {
                if (board[row][column] == EMPTY) {
                    break;
                }
                row++;
            }
            board[row][column] = currentPlayer;
            return row;
        }

        private static boolean checkWin(int row, int column) {
            return checkHorizontal(row) || checkVertical(column) || checkDiagonal(row, column);
        }

        private static boolean checkHorizontal(int row) {
            int count = 0;
            for (int i = 0; i < COLUMNS; i++) {
                if (board[row][i] == currentPlayer) {
                    count++;
                    if (count == 4) {
                        return true;
                    }
                } else {
                    count = 0;
                }
            }
            return false;
        }

        private static boolean checkVertical(int column) {
            int count = 0;
            for (int i = 0; i < ROWS; i++) {
                if (board[i][column] == currentPlayer) {
                    count++;
                    if (count == 4) {
                        return true;
                    }
                } else {
                    count = 0;
                }
            }
            return false;
        }

        private static boolean checkDiagonal(int row, int column) {
            int count = 0;
            int i = row;
            int j = column;
            while (i >= 0 && j < COLUMNS && board[i][j] == currentPlayer) {
                count++;
                if (count == 4) {
                    return true;
                }
                i--;
                j++;
            }

            i = row;
            j = column;
            while (i >= 0 && j >= 0 && board[i][j] == currentPlayer) {
                count++;
                if (count == 4) {
                    return true;
                }
                i--;
                j--;
            }
            return false;
        }

        private static boolean boardIsFull() {
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLUMNS; j++) {
                    if (board[i][j] == EMPTY) {
                        return false;
                    }
                }
            }
            return true;
        }

        private static void switchPlayer() {
            currentPlayer = (currentPlayer == RED) ? BLACK : RED;
        }
    }

    // S54.
    public static void morrisTraversal(TreeNode root) {
        TreeNode current = root;
        while (current != null) {
            if (current.left == null) {
                System.out.print(current.val + " ");
                current = current.right;
            } else {
                // Find the rightmost node in the left subtree
                TreeNode predecessor = current.left;
                while (predecessor.right != null && predecessor.right != current) {
                    predecessor = predecessor.right;
                }

                if (predecessor.right == null) {
                    // Make current the right child of its inorder predecessor
                    predecessor.right = current;
                    current = current.left;
                } else {
                    predecessor.right = null; // Restore the original tree structure
                    System.out.print(current.val + " ");
                    current = current.right;
                }
            }
        }
    }

    // S55.
    public static List<Character> getLanguageOrder(String[] words) {
        Map<Character, Set<Character>> graph = new HashMap<>();

        for (String word : words) {
            for (char c : word.toCharArray()) {
                graph.put(c, new HashSet<>());
            }
        }

        for (int i = 1; i < words.length; i++) {
            String prevWord = words[i - 1];
            String currWord = words[i];
            int minLength = Math.min(prevWord.length(), currWord.length());

            for (int j = 0; j < minLength; j++) {
                char prevChar = prevWord.charAt(j);
                char currChar = currWord.charAt(j);

                if (prevChar != currChar) {
                    graph.get(prevChar).add(currChar);
                    break;
                }
            }
        }

        // Perform topological sorting to get the order of letters
        List<Character> order = new ArrayList<>();
        Set<Character> visited = new HashSet<>();

        for (char c : graph.keySet()) {
            if (!visited.contains(c)) {
                dfs(graph, c, visited, order);
            }
        }

        return order;
    }

    private static void dfs(Map<Character, Set<Character>> graph, char c, Set<Character> visited,
            List<Character> order) {
        visited.add(c);

        for (char neighbor : graph.get(c)) {
            if (!visited.contains(neighbor)) {
                dfs(graph, neighbor, visited, order);
            }
        }

        order.add(0, c);
    }

    // S56.
    static class Edge implements Comparable<Edge> {
        int src, dest, weight;

        public Edge(int src, int dest, int weight) {
            this.src = src;
            this.dest = dest;
            this.weight = weight;
        }

        public int compareTo(Edge other) {
            return other.weight - this.weight;
        }
    }

    static class UnionFind {
        int[] parent;
        int[] rank;

        public UnionFind(int size) {
            parent = new int[size];
            rank = new int[size];
            for (int i = 0; i < size; i++) {
                parent[i] = i;
                rank[i] = 0;
            }
        }

        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);

            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }

    public static List<Edge> findMaximumSpanningTree(List<Edge> edges, int numVertices) {
        Collections.sort(edges);
        UnionFind uf = new UnionFind(numVertices);
        List<Edge> mst = new ArrayList<>();

        for (Edge edge : edges) {
            int srcParent = uf.find(edge.src);
            int destParent = uf.find(edge.dest);

            if (srcParent != destParent) {
                uf.union(srcParent, destParent);
                mst.add(edge);
            }
        }

        return mst;
    }

    // S57.
    static class MinAndMax {
        int min;
        int max;

        public MinAndMax(int min, int max) {
            this.min = min;
            this.max = max;
        }
    }

    public static MinAndMax findMinMax(int[] arr) {
        if (arr.length == 0) {
            throw new IllegalArgumentException("Array must not be empty.");
        }

        int min, max;

        if (arr[0] < arr[1]) {
            min = arr[0];
            max = arr[1];
        } else {
            min = arr[1];
            max = arr[0];
        }

        for (int i = 2; i < arr.length - 1; i += 2) {
            int num1 = arr[i];
            int num2 = arr[i + 1];

            if (num1 < num2) {
                min = Math.min(min, num1);
                max = Math.max(max, num2);
            } else {
                min = Math.min(min, num2);
                max = Math.max(max, num1);
            }
        }

        if (arr.length % 2 != 0) {
            int lastNum = arr[arr.length - 1];
            min = Math.min(min, lastNum);
            max = Math.max(max, lastNum);
        }

        return new MinAndMax(min, max);
    }

    // S58.
    public static int solveBlackjack(List<Integer> deck) {
        int wins = 0;
        int losses = 0;

        int iterations = 10000;
        for (int i = 0; i < iterations; i++) {
            int playerScore = playBlackjack(deck);
            int dealerScore = playBlackjack(deck);

            if (playerScore <= 21 && (playerScore > dealerScore || dealerScore > 21)) {
                wins++;
            } else if (playerScore > 21 || (playerScore < dealerScore && dealerScore <= 21)) {
                losses++;
            } else if (playerScore == dealerScore) {
                continue;
            }
        }

        return wins - losses;
    }

    private static int playBlackjack(List<Integer> deck) {
        Collections.shuffle(deck);
        int score = 0;
        int numAces = 0;

        for (int card : deck) {
            if (card >= 2 && card <= 10) {
                score += card;
            } else if (card >= 11 && card <= 13) {
                score += 10;
            } else if (card == 1) {
                score += 1;
                numAces++;
            }

            while (numAces > 0 && score <= 11) {
                score += 10;
                numAces--;
            }

            if (score >= 13 || score >= 21) {
                break;
            }
        }

        return score;
    }

    // S59.
    public static int minSwaps(int[] row) {
        int n = row.length;
        int swaps = 0;

        for (int i = 0; i < n; i++) {
            int partner = row[i] ^ 1;
            if (partner % 2 == 1 && row[i + 1] != partner) {
                int j = findPartnerIndex(row, i + 1, partner);
                swap(row, i + 1, j);
                swaps++;
            } else if (partner % 2 == 0 && row[i - 1] != partner) {
                int j = findPartnerIndex(row, i + 1, partner);
                swap(row, i - 1, j);
                swaps++;
            }
        }

        return swaps;
    }

    private static int findPartnerIndex(int[] row, int start, int partner) {
        for (int i = start; i < row.length; i++) {
            if (row[i] == partner) {
                return i;
            }
        }
        return -1;
    }

    // S60.
    static class SubscriberTracker {
        private int[] prefixSum;

        public SubscriberTracker(int[] subscribers) {
            this.prefixSum = new int[subscribers.length + 1];
            buildPrefixSum(subscribers);
        }

        private void buildPrefixSum(int[] subscribers) {
            prefixSum[0] = 0;
            for (int i = 1; i <= subscribers.length; i++) {
                prefixSum[i] = prefixSum[i - 1] + subscribers[i - 1];
            }
        }

        public void update(int hour, int value) {
            if (hour >= 1 && hour <= prefixSum.length) {
                while (hour < prefixSum.length) {
                    prefixSum[hour] += value;
                    hour++;
                }
            }
        }

        public int query(int start, int end) {
            if (start > 0) {
                return prefixSum[end] - prefixSum[start - 1];
            } else {
                return prefixSum[end];
            }
        }
    }

    // S61.
    public static int findMaximum(int a, int b) {
        int diff = a - b;
        int sign = (diff >> 31) & 0x1;

        int maximum = a - sign * diff;

        return maximum;
    }

    // S62.
    static class TrieNode {
        TrieNode[] children = new TrieNode[2];
    }

    static void insert(TrieNode root, int num) {
        TrieNode node = root;
        for (int i = 31; i >= 0; i--) {
            int bit = (num >> i) & 1;

            if (node.children[bit] == null) {
                node.children[bit] = new TrieNode();
            }

            node = node.children[bit];
        }
    }

    static int findMaximumXOR(int[] nums) {
        if (nums == null || nums.length == 0) {
            throw new IllegalArgumentException("Array is empty or null.");
        }

        TrieNode root = new TrieNode();

        for (int num : nums) {
            insert(root, num);
        }

        int maxXOR = Integer.MIN_VALUE;

        for (int num : nums) {
            TrieNode node = root;
            int currXOR = 0;

            for (int i = 31; i >= 0; i--) {
                int bit = (num >> i) & 1;

                // Choose the opposite bit if available in the Trie
                if (node.children[1 - bit] != null) {
                    currXOR |= (1 << i);
                    node = node.children[1 - bit];
                } else {
                    node = node.children[bit];
                }
            }

            maxXOR = Math.max(maxXOR, currXOR);
        }

        return maxXOR;
    }

    // S63.
    static class WordTrieNode {
        WordTrieNode[] children;
        boolean isEndOfWord;

        WordTrieNode() {
            children = new WordTrieNode[26];
            isEndOfWord = false;
        }
    }

    public static List<Character> findWinningStartLetters(List<String> dictionary) {
        List<Character> winningStartLetters = new ArrayList<>();
        WordTrieNode root = buildTrie(dictionary);

        for (char letter = 'a'; letter <= 'z'; letter++) {
            if (root.children[letter - 'a'] == null) {
                continue;
            }
            if (hasWinningStrategy(root, letter, 0)) {
                winningStartLetters.add(letter);
            }
        }

        return winningStartLetters;
    }

    public static boolean hasWinningStrategy(WordTrieNode node, char letter, int count) {
        int index = letter - 'a';

        count++;
        WordTrieNode childNode = node.children[index];

        if (childNode.isEndOfWord) {
            if (count % 2 == 0) {
                return true;
            }
            return false;
        }

        boolean winningInAllSubtrees = false;
        for (char nextLetter = 'a'; nextLetter <= 'z'; nextLetter++) {
            if (childNode.children[nextLetter - 'a'] == null) {
                continue;
            }
            if (hasWinningStrategy(childNode, nextLetter, count)) {
                winningInAllSubtrees = true;
            } else {
                return false;
            }
        }

        return winningInAllSubtrees;
    }

    public static WordTrieNode buildTrie(List<String> dictionary) {
        WordTrieNode root = new WordTrieNode();

        for (String word : dictionary) {
            WordTrieNode currentNode = root;

            for (char letter : word.toCharArray()) {
                int index = letter - 'a';

                if (currentNode.children[index] == null) {
                    currentNode.children[index] = new WordTrieNode();
                }

                currentNode = currentNode.children[index];
            }

            currentNode.isEndOfWord = true;
        }

        return root;
    }

    // S64.
    static Set<String> seen = new HashSet<String>();
    static Vector<Integer> edges = new Vector<Integer>();

    static String findDeBruijnSequence(int k, String C) {
        seen.clear();
        edges.clear();

        String startingNode = getStartString(k - 1, C.charAt(0));
        dfs(startingNode, C.length(), C);

        String result = "";

        int l = (int) Math.pow(C.length(), k);
        for (int i = 0; i < l; ++i)
            result += C.charAt(edges.get(i));

        return result;
    }

    static void dfs(String node, int k, String C) {
        for (int i = 0; i < k; ++i) {
            String str = node + C.charAt(i);
            if (!seen.contains(str)) {
                seen.add(str);
                dfs(str.substring(1), k, C);
                edges.add(i);
            }
        }
    }

    private static String getStartString(int n, char charAt) {
        String str = "";
        for (int i = 0; i < n; i++)
            str += charAt;
        return str;
    }

    // S65.
    private static final int BOARD_SIZE = 8;

    public static boolean isKingInCheck(char[][] chessboard) {
        int kingRow = -1;
        int kingCol = -1;

        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                if (chessboard[row][col] == 'K') {
                    kingRow = row;
                    kingCol = col;
                    break;
                }
            }
            if (kingRow != -1) {
                break;
            }
        }

        return isAttackedByRook(chessboard, kingRow, kingCol)
                || isAttackedByBishop(chessboard, kingRow, kingCol)
                || isAttackedByQueen(chessboard, kingRow, kingCol)
                || isAttackedByPawn(chessboard, kingRow, kingCol)
                || isAttackedByKnight(chessboard, kingRow, kingCol);
    }

    private static boolean isValidPosition(int row, int col) {
        return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
    }

    private static boolean isAttackedByRook(char[][] chessboard, int kingRow, int kingCol) {
        int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

        for (int[] direction : directions) {
            int row = kingRow + direction[0];
            int col = kingCol + direction[1];

            while (isValidPosition(row, col)) {
                if (chessboard[row][col] != '.') {
                    if (chessboard[row][col] == 'R' || chessboard[row][col] == 'Q') {
                        return true;
                    } else {
                        break;
                    }
                }

                row += direction[0];
                col += direction[1];
            }
        }

        return false;
    }

    private static boolean isAttackedByBishop(char[][] chessboard, int kingRow, int kingCol) {
        int[][] directions = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };

        for (int[] direction : directions) {
            int row = kingRow + direction[0];
            int col = kingCol + direction[1];

            while (isValidPosition(row, col)) {
                if (chessboard[row][col] != '.') {
                    if (chessboard[row][col] == 'B' || chessboard[row][col] == 'Q') {
                        return true;
                    } else {
                        break;
                    }
                }

                row += direction[0];
                col += direction[1];
            }
        }

        return false;
    }

    private static boolean isAttackedByQueen(char[][] chessboard, int kingRow, int kingCol) {
        return isAttackedByRook(chessboard, kingRow, kingCol) || isAttackedByBishop(chessboard, kingRow, kingCol);
    }

    private static boolean isAttackedByPawn(char[][] chessboard, int kingRow, int kingCol) {
        int[][] directions = { { -1, -1 }, { -1, 1 } };

        for (int[] direction : directions) {
            int row = kingRow + direction[0];
            int col = kingCol + direction[1];

            if (isValidPosition(row, col) && chessboard[row][col] == 'P') {
                return true;
            }
        }

        return false;
    }

    private static boolean isAttackedByKnight(char[][] chessboard, int kingRow, int kingCol) {
        int[][] moves = {
                { -2, -1 }, { -2, 1 }, { -1, -2 }, { -1, 2 },
                { 1, -2 }, { 1, 2 }, { 2, -1 }, { 2, 1 }
        };

        for (int[] move : moves) {
            int row = kingRow + move[0];
            int col = kingCol + move[1];

            if (isValidPosition(row, col) && chessboard[row][col] == 'N') {
                return true;
            }
        }

        return false;
    }

    // S66.
    public static boolean binarySearch(int[] nums, int x) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == x) {
                return true;
            } else if (nums[mid] < x) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return false;
    }

    // S67.
    public static int evaluateMathExpression(String expression) {
        Stack<Integer> operandStack = new Stack<>();
        Stack<Character> operatorStack = new Stack<>();

        int i = 0;
        while (i < expression.length()) {
            char ch = expression.charAt(i);

            if (ch == ' ') {
                i++;
                continue;
            }

            if (Character.isDigit(ch) || ch == '-') {
                int num = 0;
                int sign = 1;
                if (ch == '-') {
                    sign = -1;
                    i++;
                }
                while (i < expression.length() && Character.isDigit(expression.charAt(i))) {
                    num = num * 10 + (expression.charAt(i) - '0');
                    i++;
                }
                i--;
                operandStack.push(sign * num);
            } else if (ch == '(') {
                operatorStack.push(ch);
            } else if (ch == ')') {
                while (!operatorStack.isEmpty() && operatorStack.peek() != '(') {
                    performOperation(operandStack, operatorStack);
                }
                operatorStack.pop();
            } else if (ch == '+' || ch == '-') {
                while (!operatorStack.isEmpty() && operatorStack.peek() != '(') {
                    performOperation(operandStack, operatorStack);
                }
                operatorStack.push(ch);
            }

            i++;
        }

        while (!operatorStack.isEmpty()) {
            performOperation(operandStack, operatorStack);
        }

        return operandStack.pop();
    }

    public static void performOperation(Stack<Integer> operandStack, Stack<Character> operatorStack) {
        int operand2 = operandStack.pop();
        int operand1 = operandStack.pop();
        char operator = operatorStack.pop();

        int result;
        if (operator == '+') {
            result = operand1 + operand2;
        } else {
            result = operand1 - operand2;
        }

        operandStack.push(result);
    }

    // S68.
    static class Building {
        int left;
        int right;
        int height;

        Building(int left, int right, int height) {
            this.left = left;
            this.right = right;
            this.height = height;
        }
    }

    static class Event implements Comparable<Event> {
        int x;
        boolean isStart;
        int height;

        Event(int x, boolean isStart, int height) {
            this.x = x;
            this.isStart = isStart;
            this.height = height;
        }

        @Override
        public int compareTo(Event other) {
            if (this.x != other.x) {
                return Integer.compare(this.x, other.x);
            }

            if (this.isStart && !other.isStart) {
                return -1;
            }
            if (!this.isStart && other.isStart) {
                return 1;
            }

            if (this.isStart) {
                return Integer.compare(other.height, this.height);
            }
            return Integer.compare(this.height, other.height);
        }
    }

    public static List<int[]> getSkyline(Building[] buildings) {
        List<int[]> skyline = new ArrayList<>();

        List<Event> events = new ArrayList<>();
        for (Building building : buildings) {
            events.add(new Event(building.left, true, building.height));
            events.add(new Event(building.right, false, building.height));
        }
        Collections.sort(events);

        PriorityQueue<Integer> heights = new PriorityQueue<>(Collections.reverseOrder());
        heights.offer(0);

        int prevHeight = 0;
        for (Event event : events) {
            if (event.isStart) {
                heights.offer(event.height); // Add building height
            } else {
                heights.remove(event.height); // Remove building height
            }

            int currentHeight = heights.peek();
            if (currentHeight != prevHeight) {
                skyline.add(new int[] { event.x, currentHeight });
                prevHeight = currentHeight;
            }
        }

        return skyline;
    }

    // S69.
    public static boolean isFirstPlayerWinNim(int[] heaps) {
        int nimSum = 0;

        for (int heap : heaps) {
            nimSum ^= heap;
        }

        return nimSum != 0;
    }

    // S70.
    public static Map<Integer, Integer> divideTeams(Map<Integer, List<Integer>> students) {
        Map<Integer, Integer> teams = new HashMap<>();
        Set<Integer> visited = new HashSet<>();

        for (int student : students.keySet()) {
            if (!visited.contains(student)) {
                if (!dfs(student, students, teams, visited, 1)) {
                    return null; // Graph is not bipartite
                }
            }
        }

        return teams;
    }

    public static boolean dfs(int student, Map<Integer, List<Integer>> students, Map<Integer, Integer> teams,
            Set<Integer> visited, int team) {
        visited.add(student);
        teams.put(student, team);

        for (int enemy : students.getOrDefault(student, Collections.emptyList())) {
            if (visited.contains(enemy)) {
                if (teams.get(enemy) == team) {
                    return false;
                }
            } else {
                if (!dfs(enemy, students, teams, visited, 3 - team)) {
                    return false; // Enemy subtree is not bipartite
                }
            }
        }

        return true;
    }

    public static List<Integer> getStudentsByTeam(Map<Integer, Integer> teams, int team) {
        List<Integer> students = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : teams.entrySet()) {
            if (entry.getValue() == team) {
                students.add(entry.getKey());
            }
        }
        return students;
    }

    public static void main(String[] args) {
        /*
         * Q1.
         * Given an array of integers, return a new array such that each element at
         * index i of the new array is the product of all the numbers in the original
         * array except the one at i.
         * For example, if our input was [1, 2, 3, 4, 5], the expected output would be
         * [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would
         * be [2, 3, 6].
         * Follow-up: what if you can't use division?
         */
        System.out.println("========= Q1 ==========");
        int[] nums1 = { 1, 2, 3, 4, 5 };
        int[] result = productExceptSelf(nums1);
        for (int i : result) {
            System.out.print(i + " ");
        }
        System.out.println();

        int[] nums2 = { 3, 2, 1 };
        result = productExceptSelf(nums2);
        for (int i : result) {
            System.out.print(i + " ");
        }
        System.out.println();

        /*
         * Q2.
         * Given an array of integers, find the first missing positive integer in linear
         * time and constant space. In other words, find the lowest positive integer
         * that does not exist in the array. The array can contain duplicates and
         * negative numbers as well.
         * For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0]
         * should give 3.
         * You can modify the input array in-place.
         */
        System.out.println("========= Q2 ==========");
        int[] missingInt1 = { 3, 4, -1, 1 };
        System.out.println(findMissingPositive(missingInt1));

        int[] missingInt2 = { 1, 2, 0 };
        System.out.println(findMissingPositive(missingInt2));

        /*
         * Q3.
         * An XOR linked list is a more memory efficient doubly linked list. Instead of
         * each node holding next and prev fields, it holds a field named both, which is
         * an XOR of the next node and the previous node. Implement an XOR linked list;
         * it has an add(element) which adds the element to the end, and a get(index)
         * which returns the node at index.
         * If using a language that has no pointers (such as Python), you can assume you
         * have access to get_pointer and dereference_pointer functions that converts
         * between nodes and memory addresses.
         */
        System.out.println("========= Q3 ==========");
        XORLinkedList list = new XORLinkedList();

        list.add(10);
        list.add(20);
        list.add(30);
        list.add(40);
        list.add(50);

        XORNode node1 = list.get(0);
        XORNode node2 = list.get(2);
        XORNode node3 = list.get(4);

        System.out.println(node1.getData()); // Output: 10
        System.out.println(node2.getData()); // Output: 30
        System.out.println(node3.getData()); // Output: 50

        /*
         * Q4.
         * Given a list of integers, write a function that returns the largest sum of
         * non-adjacent numbers. Numbers can be 0 or negative.
         * For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5,
         * 1, 1, 5] should return 10, since we pick 5 and 5.
         * Follow-up: Can you do this in O(N) time and constant space?
         */
        System.out.println("========= Q4 ==========");
        int[] numsTonFindLargestNonAdjacentSum1 = { 2, 4, 6, 2, 5 };
        int[] numsTonFindLargestNonAdjacentSum2 = { 5, 1, 1, 5 };

        int result1 = largestSumNonAdjacent(numsTonFindLargestNonAdjacentSum1);
        int result2 = largestSumNonAdjacent(numsTonFindLargestNonAdjacentSum2);

        System.out.println(result1); // Output: 13
        System.out.println(result2); // Output: 10

        /*
         * Q5.
         * There exists a staircase with N steps, and you can climb up either 1 or 2
         * steps at a time. Given N, write a function that returns the number of unique
         * ways you can climb the staircase. The order of the steps matters.
         * For example, if N is 4, then there are 5 unique ways:
         * 1, 1, 1, 1
         * 2, 1, 1
         * 1, 2, 1
         * 1, 1, 2
         * 2, 2
         * What if, instead of being able to climb 1 or 2 steps at a time, you could
         * climb any number from a set of positive integers X? For example, if X = {1,
         * 3, 5}, you could climb 1, 3, or 5 steps at a time.
         */
        System.out.println("========= Q5 ==========");
        int n = 7;
        int[] X = { 1, 3, 5 };
        int ways = countWays(n, X);
        System.out.println("Number of unique ways to climb the staircase with " + n + " steps using steps from set "
                + Arrays.toString(X) + ": " + ways);

        /*
         * Q6.
         * Given an integer k and a string s, find the length of the longest substring
         * that contains at most k distinct characters.
         * For example, given s = "abcba" and k = 2, the longest substring with k
         * distinct characters is "bcb".
         */
        System.out.println("========= Q6 ==========");
        String s = "abcba";
        int k = 2;
        int length = longestSubstringWithKDistinct(s, k);
        System.out.println("Length of the longest substring with " + k + " distinct characters: " + length);

        /*
         * Q7.
         * Suppose we represent our file system by a string in the following manner:
         * The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:
         * " dir               "
         * "     subdir1       "
         * "     subdir2       "
         * "         file.ext  "
         * The directory dir contains an empty sub-directory subdir1 and a sub-directory
         * subdir2 containing a file file.ext.
         * The string
         * "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
         * represents:
         * " dir                       "
         * "     subdir1               "
         * "         file1.ext         "
         * "         subsubdir1        "
         * "     subdir2               "
         * "         subsubdir2        "
         * "             file2.ext     "
         * The directory dir contains two sub-directories subdir1 and subdir2. subdir1
         * contains a file file1.ext and an empty second-level sub-directory subsubdir1.
         * subdir2 contains a second-level sub-directory subsubdir2 containing a file
         * file2.ext.
         * We are interested in finding the longest (number of characters) absolute path
         * to a file within our file system. For example, in the second example above,
         * the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its
         * length is 32 (not including the double quotes).
         * Given a string representing the file system in the above format, return the
         * length of the longest absolute path to a file in the abstracted file system.
         * If there is no file in the system, return 0.
         * Note:
         * The name of a file contains at least a period and an extension.
         * The name of a directory or sub-directory will not contain a period.
         */
        System.out.println("========= Q7 ==========");
        String input1 = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext";
        int length1 = lengthLongestPath(input1);
        System.out.println("Length of the longest absolute path: " + length1);

        String input2 = "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext";
        int length2 = lengthLongestPath(input2);
        System.out.println("Length of the longest absolute path: " + length2);

        /*
         * Q8.
         * Given an array of integers and a number k, where 1 <= k <= length of the
         * array, compute the maximum values of each subarray of length k.
         * For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get: [10,
         * 7, 8, 8], since:
         * 10 = max(10, 5, 2)
         * 7 = max(5, 2, 7)
         * 8 = max(2, 7, 8)
         * 8 = max(7, 8, 7)
         * Do this in O(n) time and O(k) space. You can modify the input array in-place
         * and you do not need to store the results. You can simply print them out as
         * you compute them.
         */
        System.out.println("========= Q8 ==========");
        int[] nums = { 10, 5, 2, 7, 8, 7 };
        int kRange = 3;
        printMaxOfSubarrays(nums, kRange);
        System.out.println();

        /*
         * Q9.
         * Implement regular expression matching with the following special characters:
         * . (period) which matches any single character
         * * (asterisk) which matches zero or more of the preceding element
         * That is, implement a function that takes in a string and a valid regular
         * expression and returns whether or not the string matches the regular
         * expression.
         * For example, given the regular expression "ra." and the string "ray", your
         * function should return true. The same regular expression on the string
         * "raymond" should return false.
         * Given the regular expression ".*at" and the string "chat", your function
         * should return true. The same regular expression on the string "chats" should
         * return false.
         */
        System.out.println("========= Q9 ==========");
        String text = "ray";
        String pattern = "ra.";
        System.out.println(isMatch(text, pattern)); // true

        text = "raymond";
        System.out.println(isMatch(text, pattern)); // false

        text = "chat";
        pattern = ".*at";
        System.out.println(isMatch(text, pattern)); // true

        text = "chats";
        System.out.println(isMatch(text, pattern)); // false

        /*
         * Q10.
         * Suppose you are given a table of currency exchange rates, represented as a 2D
         * array. Determine whether there is a possible arbitrage: that is, whether
         * there is some sequence of trades you can make, starting with some amount A of
         * any currency, so that you can end up with some amount greater than A of that
         * currency.
         * There are no transaction costs and you can trade fractional quantities.
         */
        System.out.println("========= Q10 ==========");
        double[][] rates = {
                { 1.0, 0.85, 0.77 },
                { 1.18, 1.0, 0.91 },
                { 1.30, 1.10, 1.0 }
        };

        if (hasArbitrage(rates)) {
            System.out.println("Arbitrage is possible.");
        } else {
            System.out.println("Arbitrage is not possible.");
        }

        /*
         * Q11.
         * Given an array of strictly the characters 'R', 'G', and 'B', segregate the
         * values of the array so that all the Rs come first, the Gs come second, and
         * the Bs come last. You can only swap elements of the array.
         * Do this in linear time and in-place.
         * For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should
         * become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].
         */
        System.out.println("========= Q11 ==========");
        char[] colors = { 'G', 'B', 'R', 'R', 'B', 'R', 'G' };
        segregateColors(colors);
        System.out.println(Arrays.toString(colors));

        /*
         * Q12.
         * You have an N by N board. Write a function that, given N, returns the number
         * of possible arrangements of the board where N queens can be placed on the
         * board without threatening each other, i.e. no two queens share the same row,
         * column, or diagonal.
         */
        System.out.println("========= Q12 ==========");
        int N = 4;
        int count = countNQueens(N);
        System.out.println("Number of possible arrangements: " + count);

        /*
         * Q13.
         * Given an array of integers where every integer occurs three times except for
         * one integer, which only occurs once, find and return the non-duplicated
         * integer.
         * For example, given [6, 1, 3, 3, 3, 6, 6], return 1. Given [13, 19, 13, 13],
         * return 19.
         * Do this in O(N) time and O(1) space.
         */
        System.out.println("========= Q13 ==========");
        int[] numsToFindSingleAppearance1 = { 6, 1, 3, 3, 3, 6, 6 };
        int singleAppearanceInt = findNonDuplicated(numsToFindSingleAppearance1);
        System.out.println("Non-duplicated integer: " + singleAppearanceInt);

        int[] numsToFindSingleAppearance2 = { 13, 19, 13, 13 };
        singleAppearanceInt = findNonDuplicated(numsToFindSingleAppearance2);
        System.out.println("Non-duplicated integer: " + singleAppearanceInt);

        /*
         * Q14.
         * Given a list of integers S and a target number k, write a function that
         * returns a subset of S that adds up to k. If such a subset cannot be made,
         * then return null.
         * Integers can appear more than once in the list. You may assume all numbers in
         * the list are positive.
         * For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1]
         * since it sums up to 24.
         */
        System.out.println("========= Q1 4 ==========");
        int[] numsToFindSubset = { 12, 1, 61, 5, 9, 2 };
        int target = 24;
        List<Integer> subset = findSubset(numsToFindSubset, target);

        if (subset == null) {
            System.out.println("No subset found.");
        } else {
            System.out.println("Subset: " + subset);
        }

        /*
         * Q15.
         * Given a string, find the longest palindromic contiguous substring. If there
         * are more than one with the maximum length, return any one.
         * For example, the longest palindromic substring of "aabcdcb" is "bcdcb". The
         */
        System.out.println("========= Q15 ==========");
        String s1 = "aabcdcb";
        String s2 = "bananas";

        System.out.println("Longest Palindromic Substring 1: " + longestPalindromicSubstring(s1));
        System.out.println("Longest Palindromic Substring 2: " + longestPalindromicSubstring(s2));

        /*
         * Q16.
         * Implement an LRU (Least Recently Used) cache. It should be able to be
         * initialized with a cache size n, and contain the following methods:
         * set(key, value): sets key to value. If there are already n items in the cache
         * and we are adding a new item, then it should also remove the least recently
         * used item.
         * get(key): gets the value at key. If no such key exists, return null.
         * Each operation should run in O(1) time.
         */
        System.out.println("========= Q16 ==========");
        LRUCache cache = new LRUCache(3);

        cache.set(1, 10);
        cache.set(2, 20);
        cache.set(3, 30);

        System.out.println(cache.get(1)); // Output: 10
        System.out.println(cache.get(2)); // Output: 20

        cache.set(4, 40);

        System.out.println(cache.get(1)); // Output: -1 (not found, as it was evicted)
        System.out.println(cache.get(3)); // Output: 30
        System.out.println(cache.get(4)); // Output: 40

        /*
         * Q17.
         * Sudoku is a puzzle where you're given a partially-filled 9 by 9 grid with
         * digits. The objective is to fill the grid with the constraint that every row,
         * column, and box (3 by 3 subgrid) must contain all of the digits from 1 to 9.
         * Implement an efficient sudoku solver.
         */
        System.out.println("========= Q17 ==========");
        int[][] grid = {
                { 5, 3, 0, 0, 7, 0, 0, 0, 0 },
                { 6, 0, 0, 1, 9, 5, 0, 0, 0 },
                { 0, 9, 8, 0, 0, 0, 0, 6, 0 },
                { 8, 0, 0, 0, 6, 0, 0, 0, 3 },
                { 4, 0, 0, 8, 0, 3, 0, 0, 1 },
                { 7, 0, 0, 0, 2, 0, 0, 0, 6 },
                { 0, 6, 0, 0, 0, 0, 2, 8, 0 },
                { 0, 0, 0, 4, 1, 9, 0, 0, 5 },
                { 0, 0, 0, 0, 8, 0, 0, 7, 9 }
        };

        if (solveSudoku(grid)) {
            printGrid(grid);
        } else {
            System.out.println("No solution exists.");
        }

        /*
         * Q18.
         * Implement a file syncing algorithm for two computers over a low-bandwidth
         * network. What if we know the files in the two computers are mostly the same?
         */
        System.out.println("========= Q18 ==========");
        /*
         * 1. Establish a TCP connection between the two computers.
         * 2. Identify the files that need to be synced. Maintain list of files on each
         * computer and compare the differences.
         * 3. Use a differential syncing algorithm to transfer only different parts.
         * - Implement a mechanism to compare files to identify the differences such as
         * file hashing or timestamp comparison.
         * - When a file is modified, calculate the difference between old and new
         * version. This can be done by comparing contents of the files or using delta
         * encoding techniques.
         * 4. Transfer the differences over the low-bandwidth network. This can be done
         * by sending only the modified parts of the files or using compression
         * techniques to reduce data size.
         * 5. On the receiving computer, apply the received differences to update the
         * corresponding files. This can involve patching or merging the changes into
         * the existing files.
         * 6. Repeat synching process periodically or whenever changes are detected on
         * either computer.
         * 7. Implement error handling and recovery mechanisms to handle network
         * failures, file conflicts, etc.
         * 8. Monitor the syncing process and provide feedback to the users about the
         * progress and status of the synchronization.
         */

        /*
         * Q19.
         * A knight's tour is a sequence of moves by a knight on a chessboard such that
         * all squares are visited once.
         * Given N, write a function to return the number of knight's tours on an N by N
         * chessboard.
         */
        System.out.println("========= Q19 ==========");
        KnightsTour knightsTour = new KnightsTour();
        int numberOfTours = knightsTour.countKnightTours(5);
        System.out.println("Number of Knight's Tours on a 5x5 chessboard: " + numberOfTours);

        /*
         * Q20.
         * Implement an LFU (Least Frequently Used) cache. It should be able to be
         * initialized with a cache size n, and contain the following methods:
         * set(key, value): sets key to value. If there are already n items in the cache
         * and we are adding a new item, then it should also remove the least frequently
         * used item. If there is a tie, then the least recently used key should be
         * removed.
         * get(key): gets the value at key. If no such key exists, return null.
         * Each operation should run in O(1) time.
         */
        System.out.println("========= Q20 ==========");
        LFUCache lfuCache = new LFUCache(2);

        lfuCache.set(1, 10);
        lfuCache.set(2, 20);
        System.out.println(lfuCache.get(1)); // Output: 10

        lfuCache.set(3, 30);
        System.out.println(lfuCache.get(2)); // Output: -1
        System.out.println(lfuCache.get(3)); // Output: 30

        lfuCache.set(4, 40);
        System.out.println(lfuCache.get(1)); // Output: -1
        System.out.println(lfuCache.get(3)); // Output: 30
        System.out.println(lfuCache.get(4)); // Output: 40

        /*
         * Q21.
         * In a directed graph, each node is assigned an uppercase letter. We define a
         * path's value as the number of most frequently-occurring letter along that
         * path. For example, if a path in the graph goes through "ABACA", the value of
         * the path is 3, since there are 3 occurrences of 'A' on the path.
         * Given a graph with n nodes and m directed edges, return the largest value
         * path of the graph. If the largest value is infinite, then return null.
         * The graph is represented with a string and an edge list. The i-th character
         * represents the uppercase letter of the i-th node. Each tuple in the edge list
         * (i, j) means there is a directed edge from the i-th node to the j-th node.
         * Self-edges are possible, as well as multi-edges.
         * For example, the following input graph:
         * ABACA
         * [(0, 1),
         * (0, 2),
         * (2, 3),
         * (3, 4)]
         * Would have maximum value 3 using the path of vertices [0, 2, 3, 4], (A, A, C,
         * A).
         * The following input graph:
         * A
         * [(0, 0)]
         * Should return null, since we have an infinite loop.
         */
        System.out.println("========= Q21 ==========");
        String nodes1 = "ABACA";
        int[][] edges1 = {
                { 0, 1 },
                { 0, 2 },
                { 2, 3 },
                { 3, 4 }
        };

        DirectedGraph graph1 = new DirectedGraph(nodes1, edges1);
        Integer largestValuePath = graph1.findLargestValuePath();
        System.out.println("Largest value path: " + largestValuePath);

        String nodes2 = "A";
        int[][] edges2 = {
                { 0, 0 }
        };
        DirectedGraph graph2 = new DirectedGraph(nodes2, edges2);
        largestValuePath = graph2.findLargestValuePath();
        System.out.println("Largest value path: " + largestValuePath);

        /*
         * Q22.
         * Given an array of numbers, find the length of the longest increasing
         * subsequence in the array. The subsequence does not necessarily have to be
         * contiguous.
         * For example, given the array [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11,
         * 7, 15], the longest increasing subsequence has length 6: it is 0, 2, 6, 9,
         * 11, 15.
         */
        System.out.println("========= Q22 ==========");
        int[] numsToFindLIS = { 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 };
        int longestIncreasingSubsequenceLength = lengthOfLIS(numsToFindLIS);
        System.out.println("Length of the longest increasing subsequence: " + longestIncreasingSubsequenceLength);

        /*
         * Q23.
         * A rule looks like this:
         * A NE B
         * This means this means point A is located northeast of point B.
         * A SW C
         * means that point A is southwest of C.
         * Given a list of rules, check if the sum of the rules validate. For example:
         * A N B
         * B NE C
         * C N A
         * does not validate, since A cannot be both north and south of C.
         * A NW B
         * A N B
         * is considered valid.
         */
        System.out.println("========= Q23 ==========");
        List<String> rules1 = Arrays.asList("A N B", "B NE C", "C N A");
        boolean isValid = validateRules(rules1);
        System.out.println("Rules 1 validate: " + isValid);

        List<String> rules2 = Arrays.asList("A NW B", "A N B");
        isValid = validateRules(rules2);
        System.out.println("Rules 2 validate: " + isValid);

        /*
         * Q24.
         * We're given a hashmap associating each courseId key with a list of courseIds
         * values, which represents that the prerequisites of courseId are courseIds.
         * Return a sorted ordering of courses such that we can finish all courses.
         * Return null if there is no such ordering.
         * For example, given {'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'],
         * 'CSC100': []}, should return ['CSC100', 'CSC200', 'CSCS300'].
         */
        System.out.println("========= Q24 ==========");
        Map<String, List<String>> prerequisites = new HashMap<>();
        prerequisites.put("CSC300", Arrays.asList("CSC100", "CSC200"));
        prerequisites.put("CSC200", Arrays.asList("CSC100"));
        prerequisites.put("CSC100", new ArrayList<>());

        List<String> courseOrder = findCourseOrder(prerequisites);
        if (courseOrder == null) {
            System.out.println("No valid course ordering exists.");
        } else {
            System.out.println("Course ordering: " + courseOrder);
        }

        /*
         * Q25.
         * Given a tree, find the largest tree/subtree that is a BST.
         * Given a tree, return the size of the largest tree/subtree that is a BST.
         */
        System.out.println("========= Q25 ==========");
        TreeNode root = new TreeNode(10);
        root.left = new TreeNode(5);
        root.right = new TreeNode(15);
        root.left.left = new TreeNode(1);
        root.left.right = new TreeNode(8);
        root.right.right = new TreeNode(7);

        int largestBSTSize = largestBST(root);
        System.out.println("Size of the largest BST: " + largestBSTSize);

        /*
         * Q26.
         * Given a number represented by a list of digits, find the next greater
         * permutation of a number, in terms of lexicographic ordering. If there is not
         * greater permutation possible, return the permutation with the lowest
         * value/ordering.
         * For example, the list [1,2,3] should return [1,3,2]. The list [1,3,2] should
         * return [2,1,3]. The list [3,2,1] should return [1,2,3].
         * Can you perform the operation without allocating extra memory (disregarding
         * the input memory)?
         */
        System.out.println("========= Q26 ==========");
        int[] numsToFindNextPermutation1 = { 1, 2, 3 };
        nextPermutation(numsToFindNextPermutation1);
        System.out.println("Next permutation: " + Arrays.toString(numsToFindNextPermutation1));

        int[] numsToFindNextPermutation2 = { 1, 3, 2 };
        nextPermutation(numsToFindNextPermutation2);
        System.out.println("Next permutation: " + Arrays.toString(numsToFindNextPermutation2));

        int[] numsToFindNextPermutation3 = { 3, 2, 1 };
        nextPermutation(numsToFindNextPermutation3);
        System.out.println("Next permutation: " + Arrays.toString(numsToFindNextPermutation3));

        /*
         * Q27.
         * Given a word W and a string S, find all starting indices in S which are
         * anagrams of W.
         * For example, given that W is "ab", and S is "abxaba", return 0, 3, and 4.
         */
        System.out.println("========= Q27 ==========");
        String S = "abxaba";
        String W = "ab";
        List<Integer> indices = findAnagramIndices(S, W);
        System.out.println("Anagram indices: " + indices);

        /*
         * Q28.
         * Given a binary tree, find the lowest common ancestor (LCA) of two given nodes
         * in the tree. Assume that each node in the tree also has a pointer to its
         * parent.
         * https://en.wikipedia.org/wiki/Lowest_common_ancestor
         * According to the definition of LCA on Wikipedia: “The lowest common ancestor
         * is defined between two nodes v and w as the lowest node in T that has both v
         * and w as descendants (where we allow a node to be a descendant of itself).”
         */
        System.out.println("========= Q28 ==========");
        TreeNodeWithParent treeWithParent = new TreeNodeWithParent(3);
        TreeNodeWithParent nodeWithParent1 = new TreeNodeWithParent(5);
        TreeNodeWithParent nodeWithParent2 = new TreeNodeWithParent(1);
        TreeNodeWithParent nodeWithParent3 = new TreeNodeWithParent(6);
        TreeNodeWithParent nodeWithParent4 = new TreeNodeWithParent(2);
        TreeNodeWithParent nodeWithParent5 = new TreeNodeWithParent(0);
        TreeNodeWithParent nodeWithParent6 = new TreeNodeWithParent(8);
        TreeNodeWithParent nodeWithParent7 = new TreeNodeWithParent(7);
        TreeNodeWithParent nodeWithParent8 = new TreeNodeWithParent(4);

        treeWithParent.left = nodeWithParent1;
        treeWithParent.right = nodeWithParent2;
        nodeWithParent1.parent = treeWithParent;
        nodeWithParent2.parent = treeWithParent;

        nodeWithParent1.left = nodeWithParent3;
        nodeWithParent1.right = nodeWithParent4;
        nodeWithParent3.parent = nodeWithParent1;
        nodeWithParent4.parent = nodeWithParent1;

        nodeWithParent2.left = nodeWithParent5;
        nodeWithParent2.right = nodeWithParent6;
        nodeWithParent5.parent = nodeWithParent2;
        nodeWithParent6.parent = nodeWithParent2;

        nodeWithParent4.left = nodeWithParent7;
        nodeWithParent4.right = nodeWithParent8;
        nodeWithParent7.parent = nodeWithParent4;
        nodeWithParent8.parent = nodeWithParent4;

        TreeNodeWithParent p = nodeWithParent1;
        TreeNodeWithParent q = nodeWithParent2;

        TreeNodeWithParent lca = findLowestCommonAncestor(treeWithParent, p, q);
        if (lca != null) {
            System.out.println("Lowest Common Ancestor: " + lca.val);
        } else {
            System.out.println("Lowest Common Ancestor not found.");
        }

        /*
         * Q29.
         * Given a string and a set of delimiters, reverse the words in the string while
         * maintaining the relative order of the delimiters. For example, given
         * "hello/world:here", return "here/world:hello"
         * Follow-up: Does your solution work for the following cases:
         * "hello/world:here/", "hello//world:here"
         */
        System.out.println("========= Q29 ==========");
        String str1 = "hello/world:here";
        String str2 = "hello/world:here/";
        String str3 = "hello//world:here";
        Set<Character> delimiters = new HashSet<>();
        delimiters.add('/');
        delimiters.add(':');

        String reversed = reverseWordsWithDelimiters(str1, delimiters);
        System.out.println("Reversed string: " + reversed);
        reversed = reverseWordsWithDelimiters(str2, delimiters);
        System.out.println("Reversed string: " + reversed);
        reversed = reverseWordsWithDelimiters(str3, delimiters);
        System.out.println("Reversed string: " + reversed);

        /*
         * Q30.
         * Given two non-empty binary trees s and t, check whether tree t has exactly
         * the same structure and node values with a subtree of s. A subtree of s is a
         * tree consists of a node in s and all of this node's descendants. The tree s
         * could also be considered as a subtree of itself.
         */
        System.out.println("========= Q30 ==========");
        TreeNode binaryTreeS = new TreeNode(3);
        binaryTreeS.left = new TreeNode(4);
        binaryTreeS.right = new TreeNode(5);
        binaryTreeS.left.left = new TreeNode(1);
        binaryTreeS.left.right = new TreeNode(2);
        binaryTreeS.left.right.left = new TreeNode(0);

        TreeNode binaryTreeT = new TreeNode(4);
        binaryTreeT.left = new TreeNode(1);
        binaryTreeT.right = new TreeNode(2);

        boolean isSubtree = isSubtree(binaryTreeS, binaryTreeT);
        System.out.println("Is t a subtree of s? " + isSubtree);

        /*
         * Q31.
         * Given a string which we can delete at most k, return whether you can make a
         * palindrome.
         * For example, given 'waterrfetawx' and a k of 2, you could delete f and x to
         * get 'waterretaw'.
         */
        System.out.println("========= Q31 ==========");
        String palindromeCandidate = "waterrfetawx";
        int numOfRemoval = 2;

        boolean canMakePalindrome = canMakePalindrome(palindromeCandidate, numOfRemoval);
        System.out.println("Can make a palindrome: " + canMakePalindrome);

        /*
         * Q32.
         * Given a string, return whether it represents a number. Here are the different
         * kinds of numbers:
         * "10", a positive integer
         * "-10", a negative integer
         * "10.1", a positive real number
         * "-10.1", a negative real number
         * "1e5", a number in scientific notation
         * And here are examples of non-numbers:
         * "a"
         * "x 1"
         * "a -2"
         * "-"
         */
        System.out.println("========= Q32 ==========");
        String[] inputs = { "10", "-10", "10.1", "-10.1", "1e5", "a", "x 1", "a -2", "-" };

        for (String input : inputs) {
            boolean isNum = isNumber(input);
            System.out.println(input + " is a number: " + isNum);
        }

        /*
         * Q33.
         * Find the minimum number of coins required to make n cents.
         * You can use standard American denominations, that is, 1¢, 5¢, 10¢, and 25¢.
         * For example, given n = 16, return 3 since we can make it with a 10¢, a 5¢,
         * and a 1¢.
         */
        System.out.println("========= Q33 ==========");
        int sum = 16;
        int minimumCoins = coinChange(sum);
        System.out.println("Minimum number of coins required: " + minimumCoins);

        /*
         * Q34.
         * Implement 3 stacks using a single list:
         * " class Stack:                              "
         * "     def __init__(self):                   "
         * "         self.list = []                    "
         * 
         * "     def pop(self, stack_number):          "
         * "         pass                              "
         * 
         * "     def push(self, item, stack_number):   "
         * "         pass                              "
         */
        System.out.println("========= Q34 ==========");
        MultiStack stack = new MultiStack(10, 3);

        stack.push(1, 0);
        stack.push(2, 0);
        stack.push(3, 1);
        stack.push(4, 1);
        stack.push(5, 2);
        stack.push(6, 2);

        System.out.println("Pop from Stack 0: " + stack.pop(0));
        System.out.println("Pop from Stack 1: " + stack.pop(1));
        System.out.println("Pop from Stack 2: " + stack.pop(2));

        /*
         * Q35.
         * You're given a string consisting solely of (, ), and *. * can represent
         * either a (, ), or an empty string. Determine whether the parentheses are
         * balanced.
         * For example, (()* and (*) are balanced. )*( is not balanced.
         */
        System.out.println("========= Q35 ==========");
        String string1 = "(()*";
        String string2 = "(*)";
        String string3 = ")*(";

        System.out.println("Is '" + string1 + "' balanced? " + isBalanced(string1));
        System.out.println("Is '" + string2 + "' balanced? " + isBalanced(string2));
        System.out.println("Is '" + string3 + "' balanced? " + isBalanced(string3));

        /*
         * Q36.
         * Given a list, sort it using this method: reverse(lst, i, j), which reverses
         * lst from i to j.
         */
        System.out.println("========= Q36 ==========");
        List<Integer> lst = new ArrayList<>(List.of(9, 2, 5, 1, 7));
        System.out.println("Original List: " + lst);
        sort(lst);
        System.out.println("Sorted List: " + lst);

        /*
         * Q37.
         * Given a list of numbers L, implement a method sum(i, j) which returns the sum
         * from the sublist L[i:j] (including i, excluding j).
         * For example, given L = [1, 2, 3, 4, 5], sum(1, 3) should return sum([2, 3]),
         * which is 5.
         * You can assume that you can do some pre-processing. sum() should be optimized
         * over the pre-processing step.
         */
        System.out.println("========= Q37 ==========");
        List<Integer> L = List.of(1, 2, 3, 4, 5);
        SublistSum sublistSum = new SublistSum(L);

        System.out.println("Sum of sublist [1:3]: " + sublistSum.sum(1, 3));
        System.out.println("Sum of sublist [2:5]: " + sublistSum.sum(2, 5));
        System.out.println("Sum of sublist [0:5]: " + sublistSum.sum(0, 5));

        /*
         * Q38.
         * Given a list of points, a central point, and an integer k, find the nearest k
         * points from the central point.
         * For example, given the list of points [(0, 0), (5, 4), (3, 1)], the central
         * point (1, 2), and k = 2, return [(0, 0), (3, 1)].
         */
        System.out.println("========= Q38 ==========");
        List<Point> points = new ArrayList<>();
        points.add(new Point(0, 0));
        points.add(new Point(5, 4));
        points.add(new Point(3, 1));

        Point centralPoint = new Point(1, 2);
        int numOfPoints = 2;

        List<Point> nearestPoints = findNearestPoints(points, centralPoint, numOfPoints);

        System.out.println("Nearest " + numOfPoints + " points to (" + centralPoint.x + ", " + centralPoint.y + "):");
        for (Point point : nearestPoints) {
            System.out.println("(" + point.x + ", " + point.y + ")");
        }

        /*
         * Q39.
         * Find an efficient algorithm to find the smallest distance (measured in number
         * of words) between any two given words in a string.
         * For example, given words "hello", and "world" and a text content of
         * "dog cat hello cat dog dog hello cat world", return 1 because there's only
         * one word "cat" in between the two words.
         */
        System.out.println("========= Q39 ==========");
        String textToWordDistance = "dog cat hello cat dog dog hello cat world";
        String word1 = "hello";
        String word2 = "world";
        int smallestDistance = findSmallestDistance(textToWordDistance, word1, word2);
        System.out.println(smallestDistance); // Output: 1

        /*
         * Q40.
         * Given a tree where each edge has a weight, compute the length of the longest
         * path in the tree.
         * For example, given the following tree:
         * "    a      "
         * "   /|\     "
         * "  b c d    "
         * "     / \   "
         * "    e   f  "
         * "   / \     "
         * "  g   h    "
         * and the weights: a-b: 3, a-c: 5, a-d: 8, d-e: 2, d-f: 4, e-g: 1, e-h: 1, the
         * longest path would be c -> a -> d -> f, with a length of 17.
         * The path does not have to pass through the root, and each node can have any
         * amount of children.
         */
        System.out.println("========= Q40 ==========");
        GraphNode a = new GraphNode('a');
        GraphNode b = new GraphNode('b');
        GraphNode c = new GraphNode('c');
        GraphNode d = new GraphNode('d');
        GraphNode e = new GraphNode('e');
        GraphNode f = new GraphNode('f');
        GraphNode g = new GraphNode('g');
        GraphNode h = new GraphNode('h');

        a.edges.add(new TreeNodeEdge(b, 3));
        a.edges.add(new TreeNodeEdge(c, 5));
        a.edges.add(new TreeNodeEdge(d, 8));

        d.edges.add(new TreeNodeEdge(e, 2));
        d.edges.add(new TreeNodeEdge(f, 4));

        e.edges.add(new TreeNodeEdge(g, 1));
        e.edges.add(new TreeNodeEdge(h, 1));

        int longestPath = calculateLongestPath(a);
        System.out.println("Longest path length: " + longestPath); // Output: 17

        /*
         * Q41.
         * https://en.wikipedia.org/wiki/Reverse_Polish_notation
         * Given an arithmetic expression in Reverse Polish Notation, write a program to
         * evaluate it.
         * The expression is given as a list of numbers and operands. For example: [5,
         * 3, '+'] should return 5 + 3 = 8.
         * For example, [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']
         * should return 5, since it is equivalent to ((15 / (7 - (1 + 1))) * 3) - (2 +
         * (1 + 1)) = 5.
         * You can assume the given expression is always valid.
         */
        System.out.println("========= Q41 ==========");
        String[] expression = { "15", "7", "1", "1", "+", "-", "/", "3", "*", "2", "1", "1", "+", "+", "-" };
        int evalResult = evaluateRPN(expression);
        System.out.println("Result: " + evalResult);

        /*
         * Q42.
         * Given a list of words, find all pairs of unique indices such that the
         * concatenation of the two words is a palindrome.
         * For example, given the list ["code", "edoc", "da", "d"], return [(0, 1), (1,
         * 0), (2, 3)].
         */
        System.out.println("========= Q42 ==========");
        String[] words = { "code", "edoc", "da", "d" };
        List<List<Integer>> pairs = findPalindromePairs(words);

        for (List<Integer> pair : pairs) {
            System.out.println("(" + pair.get(0) + ", " + pair.get(1) + ")");
        }

        /*
         * Q43.
         * Alice wants to join her school's Probability Student Club. Membership dues
         * are computed via one of two simple probabilistic games.
         * The first game: roll a die repeatedly. Stop rolling once you get a five
         * followed by a six. Your number of rolls is the amount you pay, in dollars.
         * The second game: same, except that the stopping condition is a five followed
         * by a five.
         * Which of the two games should Alice elect to play? Does it even matter? Write
         * a program to simulate the two games and calculate their expected value.
         */
        System.out.println("========= Q43 ==========");
        int numSimulations = 1000000;

        double game1ExpectedValue = calculateExpectedValue(5, 6, numSimulations);
        double game2ExpectedValue = calculateExpectedValue(5, 5, numSimulations);

        System.out.println("Expected value for Game 1: " + game1ExpectedValue);
        System.out.println("Expected value for Game 2: " + game2ExpectedValue);

        if (game1ExpectedValue < game2ExpectedValue) {
            System.out.println("Alice should elect to play Game 1.");
        } else if (game2ExpectedValue < game1ExpectedValue) {
            System.out.println("Alice should elect to play Game 2.");
        } else {
            System.out.println("Alice can choose either game; they have the same expected value.");
        }

        /*
         * Q44.
         * Given a string, split it into as few strings as possible such that each
         * string is a palindrome.
         * For example, given the input string racecarannakayak, return ["racecar",
         * "anna", "kayak"].
         * Given the input string abc, return ["a", "b", "c"].
         */
        System.out.println("========= Q44 ==========");
        String inputStr1 = "racecarannakayak";
        List<String> splitResult1 = splitIntoPalindromes(inputStr1);
        System.out.println("Input: " + inputStr1);
        System.out.println("Result: " + splitResult1);

        String inputStr2 = "abc";
        List<String> splitResult2 = splitIntoPalindromes(inputStr2);
        System.out.println("Input: " + inputStr2);
        System.out.println("Result: " + splitResult2);

        /*
         * Q45.
         * Describe what happens when you type a URL into your browser and press Enter.
         */
        System.out.println("========= Q45 ==========");
        /*
         * 1. Browser parses the URL to extract different components of the URL like
         * protocol, host, port, path, query string etc.
         * 2. The browser checks its cache to find the IP address corresponding to the
         * domain name. If not found, it sends a DNS request to a DNS server to obtain
         * the IP address of the server hosting the website.
         * 3. The browser initiates a TCP connection with the server using the obtained
         * IP address and the default port for the protocol.
         * 4. The browser sends a HTTP request to the server, including the requested
         * path, query parameters, headers, and any additional data required.
         * 5. The server receives the HTTP request and processes it. This may involve
         * executing server-side scripts, accessing databases, or performing other
         * operations to generate a response.
         * 6. The server generates an HTTP response containing the requested content,
         * along with status code, headers, and any additional data.
         * 7. The browser receives the HTTP response and starts rendering the webpage.
         * It interprets the HTML, CSS, and JavaScript code to construct the visual
         * layout, apply styles, and execute any scripts.
         * 8. As the browser parses the HTML, it encounters additional resources such as
         * images, stylesheets, or scripts, referenced in the webpage. It sends separate
         * requests for each resource and starts downloading them in parallel.
         * 9. Once all the resources are downloaded and processed, the browser displays
         * the fully rendered webpage to the user, including text, image, and
         * interactive elements.
         * 10. The user can now interact with the webpage by clicking on links,
         * submitting forms, or performing other actions, which trigger additional
         * requests and responses.
         */

        /*
         * Q46.
         * Given an array of positive integers, divide the array into two subsets such
         * that the difference between the sum of the subsets is as small as possible.
         * For example, given [5, 10, 15, 20, 25], return the sets {10, 25} and {5, 15,
         * 20}, which has a difference of 5, which is the smallest possible difference.
         */
        System.out.println("========= Q46 ==========");
        int[] numsToDivide = { 5, 10, 15, 20, 25 };
        int minDiff = minSubsetSumDifference(numsToDivide);
        System.out.println("Minimum subset sum difference: " + minDiff);

        /*
         * Q47.
         * Given a array of numbers representing the stock prices of a company in
         * chronological order, write a function that calculates the maximum profit you
         * could have made from buying and selling that stock. You're also given a
         * number fee that represents a transaction fee for each buy and sell
         * transaction.
         * You must buy before you can sell the stock, but you can make as many
         * transactions as you like.
         * For example, given [1, 3, 2, 8, 4, 10] and fee = 2, you should return 9,
         * since you could buy the stock at 1 dollar, and sell at 8 dollars, and then
         * buy it at 4 dollars and sell it at 10 dollars. Since we did two transactions,
         * there is a 4 dollar fee, so we have 7 + 6 = 13 profit minus 4 dollars of
         * fees.
         */
        System.out.println("========= Q47 ==========");
        int[] prices = { 1, 3, 2, 8, 4, 10 };
        int fee = 2;
        int maxProfitWithFee = maxProfit(prices, fee);
        System.out.println("Maximum profit: " + maxProfitWithFee);

        /*
         * Q48.
         * Let A be an N by M matrix in which every row and every column is sorted.
         * Given i1, j1, i2, and j2, compute the number of elements of M smaller than
         * M[i1, j1] and larger than M[i2, j2].
         * For example, given the following matrix:
         * [[1, 3, 7, 10, 15, 20],
         * [2, 6, 9, 14, 22, 25],
         * [3, 8, 10, 15, 25, 30],
         * [10, 11, 12, 23, 30, 35],
         * [20, 25, 30, 35, 40, 45]]
         * And i1 = 1, j1 = 1, i2 = 3, j2 = 3, return 15 as there are 15 numbers in the
         * matrix smaller than 6 or greater than 23.
         */
        System.out.println("========= Q48 ==========");
        int[][] matrix = {
                { 1, 3, 7, 10, 15, 20 },
                { 2, 6, 9, 14, 22, 25 },
                { 3, 8, 10, 15, 25, 30 },
                { 10, 11, 12, 23, 30, 35 },
                { 20, 25, 30, 35, 40, 45 }
        };

        int i1 = 1, j1 = 1, i2 = 3, j2 = 3;
        int elementCount = countElements(matrix, i1, j1, i2, j2);
        System.out.println("Number of elements: " + elementCount);

        /*
         * Q49.
         * Given a string of parentheses, find the balanced string that can be produced
         * from it using the minimum number of insertions and deletions. If there are
         * multiple solutions, return any of them.
         * For example, given "(()", you could return "(())". Given "))()(", you could
         * return "()()()()".
         */
        System.out.println("========= Q49 ==========");
        String parentheses1 = "(()";
        String balanced1 = balanceParentheses(parentheses1);
        System.out.println("Balanced string for '" + parentheses1 + "': " + balanced1);

        String parentheses2 = "))()(";
        String balanced2 = balanceParentheses(parentheses2);
        System.out.println("Balanced string for '" + parentheses2 + "': " + balanced2);

        /*
         * Q50.
         * Let X be a set of n intervals on the real line. We say that a set of points P
         * "stabs" X if every interval in X contains at least one point in P. Compute
         * the smallest set of points that stabs X.
         * For example, given the intervals [(1, 4), (4, 5), (7, 9), (9, 12)], you
         * should return [4, 9].
         */
        System.out.println("========= Q50 ==========");
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 4),
                new Interval(4, 5),
                new Interval(7, 9),
                new Interval(9, 12));

        List<Integer> stabPoints = findStabPoints(intervals);
        System.out.println("Smallest set of stab points: " + stabPoints);

        /*
         * Q51.
         * Write a program that computes the length of the longest common subsequence of
         * three given strings. For example, given "epidemiologist", "refrigeration",
         * and "supercalifragilisticexpialodocious", it should return 5, since the
         * longest common subsequence is "eieio".
         */
        System.out.println("========= Q51 ==========");
        String text1 = "epidemiologist";
        String text2 = "refrigeration";
        String text3 = "supercalifragilisticexpialodocious";

        int commonSubstringLength = longestCommonSubsequenceLength(text1, text2, text3);
        System.out.println("Length of the longest common subsequence: " + commonSubstringLength);

        /*
         * Q52.
         * We say a number is sparse if there are no adjacent ones in its binary
         * representation. For example, 21 (10101) is sparse, but 22 (10110) is not. For
         * a given input N, find the smallest sparse number greater than or equal to N.
         * Do this in faster than O(N log N) time.
         */
        System.out.println("========= Q52 ==========");
        int nonSparseNum = 22;
        int sparseNumber = findSparseNumber(nonSparseNum);
        System.out.println("Smallest sparse number greater than or equal to " + nonSparseNum + ": " + sparseNumber);

        /*
         * Q53.
         * Connect 4 is a game where opponents take turns dropping red or black discs
         * into a 7 x 6 vertically suspended grid. The game ends either when one player
         * creates a line of four consecutive discs of their color (horizontally,
         * vertically, or diagonally), or when there are no more spots left in the grid.
         * Design and implement Connect 4.
         */
        System.out.println("========= Q53 ==========");
        Connect4.initializeGame();
        Connect4.playGame();

        /*
         * Q54.
         * Typically, an implementation of in-order traversal of a binary tree has O(h)
         * space complexity, where h is the height of the tree. Write a program to
         * compute the in-order traversal of a binary tree using O(1) space.
         */
        System.out.println("========= Q54 ==========");
        TreeNode rootForInorderTraversal = new TreeNode(1);
        rootForInorderTraversal.left = new TreeNode(2);
        rootForInorderTraversal.right = new TreeNode(3);
        rootForInorderTraversal.left.left = new TreeNode(4);
        rootForInorderTraversal.left.right = new TreeNode(5);

        morrisTraversal(rootForInorderTraversal);
        System.out.println();

        /*
         * Q55.
         * You come across a dictionary of sorted words in a language you've never seen
         * before. Write a program that returns the correct order of letters in this
         * language.
         * For example, given ['xww', 'wxyz', 'wxyw', 'ywx', 'ywz'], you should return
         * ['x', 'z', 'w', 'y'].
         */
        System.out.println("========= Q55 ==========");
        String[] wordsFromNewLang = { "xww", "wxyz", "wxyw", "ywx", "ywz" };
        List<Character> order = getLanguageOrder(wordsFromNewLang);

        System.out.println("Correct order of letters in the language:");
        for (char ch : order) {
            System.out.print(ch + " ");
        }
        System.out.println();

        /*
         * Q56.
         * Recall that the minimum spanning tree is the subset of edges of a tree that
         * connect all its vertices with the smallest possible total edge weight. Given
         * an undirected graph with weighted edges, compute the maximum weight spanning
         * tree.
         */
        System.out.println("========= Q56 ==========");
        List<Edge> edges = new ArrayList<>();
        edges.add(new Edge(0, 1, 10));
        edges.add(new Edge(0, 2, 6));
        edges.add(new Edge(0, 3, 5));
        edges.add(new Edge(1, 3, 15));
        edges.add(new Edge(2, 3, 4));

        int numVertices = 4;

        List<Edge> mst = findMaximumSpanningTree(edges, numVertices);

        System.out.println("Edges of the maximum weight spanning tree:");
        for (Edge edge : mst) {
            System.out.println(edge.src + " -- " + edge.dest + " : " + edge.weight);
        }

        /*
         * Q57.
         * Given an array of numbers of length N, find both the minimum and maximum
         * using less than 2 * (N - 2) comparisons.
         */
        System.out.println("========= Q57 ==========");
        int[] arr = { 5, 7, 1, 3, 9, 2 };
        MinAndMax minAndMax = findMinMax(arr);
        System.out.println("Minimum: " + minAndMax.min);
        System.out.println("Maximum: " + minAndMax.max);

        /*
         * Q58.
         * https://en.wikipedia.org/wiki/Blackjack
         * Blackjack is a two player card game whose rules are as follows:
         * The player and then the dealer are each given two cards.
         * The player can then "hit", or ask for arbitrarily many additional cards, so
         * long as their total does not exceed 21.
         * The dealer must then hit if their total is 16 or lower, otherwise pass.
         * Finally, the two compare totals, and the one with the greatest sum not
         * exceeding 21 is the winner.
         * For this problem, cards values are counted as follows: each card between 2
         * and 10 counts as their face value, face cards count as 10, and aces count as
         * 1.
         * Given perfect knowledge of the sequence of cards in the deck, implement a
         * blackjack solver that maximizes the player's score (that is, wins minus
         * losses).
         */
        System.out.println("========= Q58 ==========");
        List<Integer> deck = new ArrayList<>();
        for (int i = 1; i <= 13; i++) {
            deck.add(i);
        }

        int score = solveBlackjack(deck);
        System.out.println("Score: " + score);

        /*
         * Q59.
         * There are N couples sitting in a row of length 2 * N. They are currently
         * ordered randomly, but would like to rearrange themselves so that each
         * couple's partners can sit side by side.
         * What is the minimum number of swaps necessary for this to happen?
         */
        System.out.println("========= Q59 ==========");
        int[] row = { 0, 3, 2, 1, 4, 7, 6, 5 };
        int minSwaps = minSwaps(row);
        System.out.println("Minimum number of swaps: " + minSwaps);

        /*
         * Q60.
         * You are given an array of length 24, where each element represents the number
         * of new subscribers during the corresponding hour. Implement a data structure
         * that efficiently supports the following:
         * update(hour: int, value: int): Increment the element at index hour by value.
         * query(start: int, end: int): Retrieve the number of subscribers that have
         * signed up between start and end (inclusive).
         * You can assume that all values get cleared at the end of the day, and that
         * you will not be asked for start and end values that wrap around midnight.
         */
        System.out.println("========= Q60 ==========");
        int[] subscribers = { 5, 3, 7, 2, 8, 4, 10, 6, 15, 9, 11, 5, 14, 7, 13, 8, 12, 6, 9, 10, 7, 11, 5, 4 };
        SubscriberTracker tracker = new SubscriberTracker(subscribers);
        tracker.update(5, 10);
        tracker.update(10, 5);
        tracker.update(15, 8);

        System.out.println(tracker.query(1, 24)); // Output: 214
        System.out.println(tracker.query(5, 15)); // Output: 125
        System.out.println(tracker.query(10, 18)); // Output: 98

        /*
         * Q61.
         * Find the maximum of two numbers without using any if-else statements,
         * branching, or direct comparisons.
         */
        System.out.println("========= Q61 ==========");
        int num1 = 10;
        int num2 = 20;

        int maximum = findMaximum(num1, num2);
        System.out.println("The maximum of " + num1 + " and " + num2 + " is: " + maximum);

        /*
         * Q62.
         * Given an array of integers, find the maximum XOR of any two elements.
         */
        System.out.println("========= Q62 ==========");
        int[] numsToFindMaxXOR = { 3, 10, 5, 25, 2, 8 };
        int maximumXOR = findMaximumXOR(numsToFindMaxXOR);
        System.out.println("Maximum XOR: " + maximumXOR);

        /*
         * Q63.
         * Ghost is a two-person word game where players alternate appending letters to
         * a word. The first person who spells out a word, or creates a prefix for which
         * there is no possible continuation, loses. Here is a sample game:
         * Player 1: g
         * Player 2: h
         * Player 1: o
         * Player 2: s
         * Player 1: t [loses]
         * Given a dictionary of words, determine the letters the first player should
         * start with, such that with optimal play they cannot lose.
         * For example, if the dictionary is ["cat", "calf", "dog", "bear"], the only
         * winning start letter would be b.
         */
        System.out.println("========= Q63 ==========");
        String[] dictionary = { "cat", "calf", "dog", "bear" };
        List<String> dictionaryList = Arrays.asList(dictionary);

        List<Character> winningStartLetters = findWinningStartLetters(dictionaryList);
        System.out.println("Winning Start Letters: " + winningStartLetters);

        /*
         * Q64.
         * Given a set of characters C and an integer k, a De Bruijn sequence is a
         * cyclic sequence in which every possible k-length string of characters in C
         * occurs exactly once.
         * For example, suppose C = {0, 1} and k = 3. Then our sequence should contain
         * the substrings {'000', '001', '010', '011', '100', '101', '110', '111'}, and
         * one possible solution would be 00010111.
         * Create an algorithm that finds a De Bruijn sequence.
         */
        System.out.println("========= Q64 ==========");
        int lengthK = 3;
        String C = "01";

        String deBruijnSequence = findDeBruijnSequence(lengthK, C);
        System.out.println("De Bruijn Sequence: " + deBruijnSequence);

        /*
         * Q65.
         * You are presented with an 8 by 8 matrix representing the positions of pieces
         * on a chess board. The only pieces on the board are the black king and various
         * white pieces. Given this matrix, determine whether the king is in check.
         * For details on how each piece moves, see here.
         * https://en.wikipedia.org/wiki/Chess_piece#Moves_of_the_pieces
         * For example, given the following matrix:
         * ...K....
         * ........
         * .B......
         * ......P.
         * .......R
         * ..N.....
         * ........
         * .....Q..
         * You should return True, since the bishop is attacking the king diagonally.
         */
        System.out.println("========= Q65 ==========");
        char[][] chessboard = {
                { '.', '.', '.', 'K', '.', '.', '.', '.' },
                { '.', '.', '.', '.', '.', '.', '.', '.' },
                { '.', 'B', '.', '.', '.', '.', '.', '.' },
                { '.', '.', '.', '.', '.', '.', 'P', '.' },
                { '.', '.', '.', '.', '.', '.', '.', 'R' },
                { '.', '.', 'N', '.', '.', '.', '.', '.' },
                { '.', '.', '.', '.', '.', '.', '.', '.' },
                { '.', '.', '.', '.', '.', 'Q', '.', '.' }
        };

        boolean isInCheck = isKingInCheck(chessboard);
        System.out.println("Is the king in check? " + isInCheck);

        /*
         * Q66.
         * Given a sorted list of integers of length N, determine if an element x is in
         * the list without performing any multiplication, division, or bit-shift
         * operations.
         * Do this in O(log N) time.
         */
        System.out.println("========= Q66 ==========");
        int[] sortedNums = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        int x = 7;

        boolean isPresent = binarySearch(sortedNums, x);
        System.out.println("Is " + x + " present? " + isPresent);

        /*
         * Q67.
         * Given a string consisting of parentheses, single digits, and positive and
         * negative signs, convert the string into a mathematical expression to obtain
         * the answer.
         * Don't use eval or a similar built-in parser.
         * For example, given '-1 + (2 + 3)', you should return 4.
         */
        System.out.println("========= Q67 ==========");
        String mathExpression = "-1 + (2 + 3)";

        int answer = evaluateMathExpression(mathExpression);
        System.out.println("Result: " + answer);

        /*
         * Q68.
         * The skyline of a city is composed of several buildings of various widths and
         * heights, possibly overlapping one another when viewed from a distance. We can
         * represent the buildings using an array of (left, right, height) tuples, which
         * tell us where on an imaginary x-axis a building begins and ends, and how tall
         * it is. The skyline itself can be described by a list of (x, height) tuples,
         * giving the locations at which the height visible to a distant observer
         * changes, and each new height.
         * Given an array of buildings as described above, create a function that
         * returns the skyline.
         * For example, suppose the input consists of the buildings [(0, 15, 3), (4, 11,
         * 5), (19, 23, 4)]. In aggregate, these buildings would create a skyline that
         * looks like the one below.
         * "      ______               "
         * "     |      |        ___   "
         * "  ___|      |___    |   |  "
         * " |   |   B  |   |   | C |  "
         * " | A |      | A |   |   |  "
         * " |   |      |   |   |   |  "
         * " ------------------------  "
         * As a result, your function should return [(0, 3), (4, 5), (11, 3), (15, 0),
         * (19, 4), (23, 0)].
         */
        System.out.println("========= Q68 ==========");
        Building[] buildings = {
                new Building(0, 15, 3),
                new Building(4, 11, 5),
                new Building(19, 23, 4)
        };

        List<int[]> skyline = getSkyline(buildings);
        for (int[] point : skyline) {
            System.out.println(Arrays.toString(point));
        }

        /*
         * Q69.
         * The game of Nim is played as follows. Starting with three heaps, each
         * containing a variable number of items, two players take turns removing one or
         * more items from a single pile. The player who eventually is forced to take
         * the last stone loses. For example, if the initial heap sizes are 3, 4, and 5,
         * a game could be played as shown below:
         * "   A  |  B  |  C   "
         * " ----------------- "
         * "   3  |  4  |  5   "
         * "   3  |  1  |  3   "
         * "   3  |  1  |  3   "
         * "   0  |  1  |  3   "
         * "   0  |  1  |  0   "
         * "   0  |  0  |  0   "
         * In other words, to start, the first player takes three items from pile B. The
         * second player responds by removing two stones from pile C. The game continues
         * in this way until player one takes last stone and loses.
         * Given a list of non-zero starting values [a, b, c], and assuming optimal
         * play, determine whether the first player has a forced win.
         */
        System.out.println("========= Q69 ==========");
        int[] heaps = { 3, 4, 5 };

        boolean isFirstPlayerWin = isFirstPlayerWinNim(heaps);
        System.out.println("Does the first player have a forced win? " + isFirstPlayerWin);

        /*
         * Q70.
         * A teacher must divide a class of students into two teams to play dodgeball.
         * Unfortunately, not all the kids get along, and several refuse to be put on
         * the same team as that of their enemies.
         * Given an adjacency list of students and their enemies, write an algorithm
         * that finds a satisfactory pair of teams, or returns False if none exists.
         * For example, given the following enemy graph you should return the teams {0,
         * 1, 4, 5} and {2, 3}.
         * " students = {       "
         * "     0: [3],        "
         * "     1: [2],        "
         * "     2: [1, 4],     "
         * "     3: [0, 4, 5],  "
         * "     4: [2, 3],     "
         * "     5: [3]         "
         * " }                  "
         * On the other hand, given the input below, you should return False.
         * " students = {          "
         * "     0: [3],           "
         * "     1: [2],           "
         * "     2: [1, 3, 4],     "
         * "     3: [0, 2, 4, 5],  "
         * "     4: [2, 3],        "
         * "     5: [3]            "
         * " }                     "
         */
        System.out.println("========= Q70 ==========");
        Map<Integer, List<Integer>> students1 = new HashMap<>();
        students1.put(0, Arrays.asList(3));
        students1.put(1, Arrays.asList(2));
        students1.put(2, Arrays.asList(1, 4));
        students1.put(3, Arrays.asList(0, 4, 5));
        students1.put(4, Arrays.asList(2, 3));
        students1.put(5, Arrays.asList(3));

        Map<Integer, Integer> teams1 = divideTeams(students1);
        if (teams1 == null) {
            System.out.println("False");
        } else {
            System.out.println("Team 1: " + getStudentsByTeam(teams1, 1));
            System.out.println("Team 2: " + getStudentsByTeam(teams1, 2));
        }

        Map<Integer, List<Integer>> students2 = new HashMap<>();
        students2.put(0, Arrays.asList(3));
        students2.put(1, Arrays.asList(2));
        students2.put(2, Arrays.asList(1, 3, 4));
        students2.put(3, Arrays.asList(0, 2, 4, 5));
        students2.put(4, Arrays.asList(2, 3));
        students2.put(5, Arrays.asList(3));

        Map<Integer, Integer> teams2 = divideTeams(students2);
        if (teams2 == null) {
            System.out.println("False");
        } else {
            System.out.println("Team 1: " + getStudentsByTeam(teams2, 1));
            System.out.println("Team 2: " + getStudentsByTeam(teams2, 2));
        }
    }

}
