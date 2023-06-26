package Hard;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

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

    }

}