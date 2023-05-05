import java.util.Set;
import java.util.HashSet;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Scanner;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

public class QuestionsAndSolutions {
    // S1.
    public static boolean doesTwoSumHaveK(int[] nums, int k) {
        Set<Integer> set = new HashSet<Integer>();
        for (int num : nums) {
            if (set.contains(k - num)) {
                return true;
            }
            set.add(num);
        }
        return false;
    }

    // S2.
    static class TreeNode<T> {
        T val;
        TreeNode<T> left;
        TreeNode<T> right;

        TreeNode(T val) {
            this.val = val;
        }
    }

    static class Tuple {
        boolean isUnival;
        int value;
        int count;

        public Tuple(boolean isUnival, int value, int count) {
            this.isUnival = isUnival;
            this.value = value;
            this.count = count;
        }
    }

    static class UnivalTreeCount {
        public int countUnivalSubtrees(TreeNode<Integer> root) {
            return helper(root).count;
        }

        private Tuple helper(TreeNode<Integer> node) {
            if (node == null) {
                return new Tuple(true, 0, 0);
            }

            Tuple left = helper(node.left);
            Tuple right = helper(node.right);

            int count = left.count + right.count;
            boolean isUnival = true;

            if (!left.isUnival || !right.isUnival) {
                isUnival = false;
            }

            if (node.left != null && node.left.val != node.val) {
                isUnival = false;
            }

            if (node.right != null && node.right.val != node.val) {
                isUnival = false;
            }

            if (isUnival) {
                count++;
            }

            return new Tuple(isUnival, node.val, count);
        }
    }

    // S3.
    static class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
        }
    }

    static class LinkedListIntersection {
        public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            int lengthA = getLength(headA);
            int lengthB = getLength(headB);

            int diff = Math.abs(lengthA - lengthB);

            ListNode ptrA = headA;
            ListNode ptrB = headB;

            if (lengthA > lengthB) {
                for (int i = 0; i < diff; i++) {
                    ptrA = ptrA.next;
                }
            } else {
                for (int i = 0; i < diff; i++) {
                    ptrB = ptrB.next;
                }
            }

            while (ptrA != null && ptrB != null) {
                if (ptrA.val == ptrB.val) {
                    return ptrA;
                }
                ptrA = ptrA.next;
                ptrB = ptrB.next;
            }

            return null;
        }

        private int getLength(ListNode node) {
            int length = 0;
            while (node != null) {
                length++;
                node = node.next;
            }
            return length;
        }
    }

    // S4.
    static class MinimumRooms {
        public int minMeetingRooms(int[][] intervals) {
            if (intervals == null || intervals.length == 0) {
                return 0;
            }

            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);

            PriorityQueue<Integer> endTimes = new PriorityQueue<>();

            endTimes.offer(intervals[0][1]);

            for (int i = 1; i < intervals.length; i++) {
                int[] interval = intervals[i];

                // If the current interval's start time is greater than or equal to the minimum
                // end time
                // in the heap, we can reuse the same room and update the minimum end time in
                // the heap
                if (interval[0] >= endTimes.peek()) {
                    endTimes.poll();
                }

                // Add the current interval's end time to the heap
                endTimes.offer(interval[1]);
            }

            return endTimes.size();
        }
    }

    // S5.
    static class Cell {
        int row;
        int col;

        Cell(int row, int col) {
            this.row = row;
            this.col = col;
        }
    }

    static class Node {
        Cell cell;
        int dist;
        Node parent;

        Node(Cell cell, int dist, Node parent) {
            this.cell = cell;
            this.dist = dist;
            this.parent = parent;
        }
    }

    static class MatrixPathFinder {
        public Integer findShortestPath(boolean[][] board, Cell start, Cell end) {
            int m = board.length;
            int n = board[0].length;
            boolean[][] visited = new boolean[m][n];
            Queue<Node> queue = new LinkedList<>();
            queue.add(new Node(start, 0, null));
            visited[start.row][start.col] = true;
            while (!queue.isEmpty()) {
                Node curr = queue.remove();
                Cell cell = curr.cell;
                if (cell.row == end.row && cell.col == end.col) {
                    return curr.dist;
                }
                int[][] directions = { { -1, 0 }, { 0, -1 }, { 1, 0 }, { 0, 1 } };
                for (int[] dir : directions) {
                    int newRow = cell.row + dir[0];
                    int newCol = cell.col + dir[1];
                    if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && !board[newRow][newCol]
                            && !visited[newRow][newCol]) {
                        queue.add(new Node(new Cell(newRow, newCol), curr.dist + 1, curr));
                        visited[newRow][newCol] = true;
                    }
                }
            }
            return null;
        }
    }

    // S6.
    class OrderLog {
        private int[] buffer;
        private int start;
        private int size;

        public OrderLog(int N) {
            buffer = new int[N];
            start = 0;
            size = 0;
        }

        public void record(int orderId) {
            buffer[(start + size) % buffer.length] = orderId;
            if (size < buffer.length) {
                size++;
            } else {
                // When the buffer is full, use start to circulate the buffer array
                start = (start + 1) % buffer.length;
            }
        }

        public int getLast(int i) {
            if (i <= 0 || i > size) {
                throw new IllegalArgumentException("Invalid i value");
            }
            return buffer[(start + size - i) % buffer.length];
        }
    }

    // S7.
    static class BracketChecker {
        public boolean isBalanced(String str) {
            Stack<Character> stack = new Stack<>();
            for (char ch : str.toCharArray()) {
                if (ch == '(' || ch == '{' || ch == '[') {
                    stack.push(ch);
                } else if (ch == ')' || ch == '}' || ch == ']') {
                    if (stack.isEmpty()) {
                        return false;
                    }
                    char top = stack.pop();
                    if ((ch == ')' && top != '(') || (ch == '}' && top != '{') || (ch == ']' && top != '[')) {
                        return false;
                    }
                }
            }
            return stack.isEmpty();
        }
    }

    // S8.
    static class RunLengthEncoderDecoder {
        public String encode(String input) {
            StringBuilder sb = new StringBuilder();
            int count = 1;
            char prevChar = input.charAt(0);
            for (int i = 1; i < input.length(); i++) {
                char currChar = input.charAt(i);
                if (currChar == prevChar) {
                    count++;
                } else {
                    sb.append(count).append(prevChar);
                    count = 1;
                    prevChar = currChar;
                }
            }
            sb.append(count).append(prevChar);
            return sb.toString();
        }

        public String decode(String input) {
            StringBuilder sb = new StringBuilder();
            int count = 0;
            for (char ch : input.toCharArray()) {
                if (Character.isDigit(ch)) {
                    count = count * 10 + Character.getNumericValue(ch);
                } else {
                    for (int i = 0; i < count; i++) {
                        sb.append(ch);
                    }
                    count = 0;
                }
            }
            return sb.toString();
        }
    }

    // S9.
    public static int editDistance(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];

        for (int i = 0; i <= s1.length(); i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= s2.length(); j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1]; // Same distance from both -1 position
                } else {
                    dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], Math.min(dp[i][j - 1], dp[i - 1][j])); // Get the minimum
                                                                                                     // of previous and
                                                                                                     // add 1 distance
                }
            }
        }

        return dp[s1.length()][s2.length()];
    }

    // S10.
    static class RunningMedian {
        private PriorityQueue<Integer> maxHeap;
        private PriorityQueue<Integer> minHeap;

        public RunningMedian() {
            maxHeap = new PriorityQueue<>(Collections.reverseOrder());
            minHeap = new PriorityQueue<>();
        }

        public void addNumber(int num) {
            // Add num to maxHeap if num is equal or less than current heap
            if (maxHeap.isEmpty() || num <= maxHeap.peek()) {
                maxHeap.offer(num);
            } else {
                minHeap.offer(num);
            }

            // Balance two heaps to find the middle
            if (maxHeap.size() - minHeap.size() > 1) {
                minHeap.offer(maxHeap.poll());
            } else if (minHeap.size() - maxHeap.size() > 1) {
                maxHeap.offer(minHeap.poll());
            }
        }

        public double getMedian() {
            if (maxHeap.isEmpty() && minHeap.isEmpty()) {
                throw new IllegalStateException("No numbers added yet.");
            }

            if (maxHeap.size() == minHeap.size()) {
                return (maxHeap.peek() + minHeap.peek()) / 2.0;
            } else if (maxHeap.size() > minHeap.size()) {
                return maxHeap.peek();
            } else {
                return minHeap.peek();
            }
        }
    }

    // S11.
    class PowerSet {
        public static Set<Set<Integer>> generatePowerSet(Set<Integer> set) {
            Set<Set<Integer>> powerSet = new HashSet<>();
            powerSet.add(new HashSet<>());

            for (int element : set) {
                // Each iteration returns new Sets with element
                Set<Set<Integer>> newSubsets = new HashSet<>();
                for (Set<Integer> subset : powerSet) {
                    Set<Integer> newSubset = new HashSet<>(subset);
                    // Additional Set with new element
                    newSubset.add(element);
                    newSubsets.add(newSubset);
                }
                powerSet.addAll(newSubsets);
            }

            return powerSet;
        }
    }

    // S12.
    static class MaxStack {
        // Additional stack to manage max value
        private Stack<Integer> stack;
        private Stack<Integer> maxStack;

        public MaxStack() {
            stack = new Stack<>();
            maxStack = new Stack<>();
        }

        public void push(int val) {
            stack.push(val);
            if (maxStack.isEmpty() || val >= maxStack.peek()) {
                maxStack.push(val);
            }
        }

        public int pop() {
            if (stack.isEmpty()) {
                throw new RuntimeException("Stack is empty");
            }
            int val = stack.pop();
            if (val == maxStack.peek()) {
                maxStack.pop();
            }
            return val;
        }

        public int max() {
            if (maxStack.isEmpty()) {
                throw new RuntimeException("Stack is empty");
            }
            return maxStack.peek();
        }
    }

    // S13.
    public static int maxProfit(int[] prices) {
        if (prices == null || prices.length < 2) {
            return 0;
        }
        int minPrice = prices[0];
        int maxProfit = 0;
        for (int i = 1; i < prices.length; i++) {
            maxProfit = Math.max(maxProfit, prices[i] - minPrice);
            minPrice = Math.min(minPrice, prices[i]);
        }
        return maxProfit;
    }

    // S14.
    public static int evaluate(TreeNode<Character> root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return Character.getNumericValue(root.val); // leaf node
        }
        int leftValue = evaluate(root.left);
        int rightValue = evaluate(root.right);
        switch (root.val) {
            case '+':
                return leftValue + rightValue;
            case '-':
                return leftValue - rightValue;
            case '*':
                return leftValue * rightValue;
            case '/':
                return leftValue / rightValue;
            default:
                throw new IllegalArgumentException("Invalid operator: " + root.val);
        }
    }

    // S15.
    static class UrlShortener {
        private Map<String, String> urlsToKeys;
        private Map<String, String> keysToUrls;
        private static final String BASE_URL = "https://example.com/";

        public UrlShortener() {
            urlsToKeys = new HashMap<>();
            keysToUrls = new HashMap<>();
        }

        public String shorten(String url) {
            if (urlsToKeys.containsKey(url)) {
                return BASE_URL + urlsToKeys.get(url);
            }
            String key = generateKey();
            urlsToKeys.put(url, key);
            keysToUrls.put(key, url);
            return BASE_URL + key;
        }

        public String restore(String shortUrl) {
            String key = shortUrl.substring(BASE_URL.length());
            if (keysToUrls.containsKey(key)) {
                return keysToUrls.get(key);
            }
            return null;
        }

        private String generateKey() {
            String key = "";
            String characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            Random random = new Random();
            for (int i = 0; i < 6; i++) {
                key += characters.charAt(random.nextInt(characters.length()));
            }
            return key;
        }
    }

    // S16.
    public static boolean isWordPresent(char[][] matrix, String target) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            String rowStr = new String(matrix[i]);
            if (rowStr.contains(target)) {
                return true;
            }
        }

        for (int j = 0; j < cols; j++) {
            StringBuilder colStrBuilder = new StringBuilder();
            for (int i = 0; i < rows; i++) {
                colStrBuilder.append(matrix[i][j]);
            }
            String colStr = colStrBuilder.toString();
            if (colStr.contains(target)) {
                return true;
            }
        }
        return false;
    }

    // S17.
    public static void printMatrixInSpiralOrder(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int topRow = 0, bottomRow = rows - 1, leftCol = 0, rightCol = cols - 1;

        while (topRow <= bottomRow && leftCol <= rightCol) {
            for (int j = leftCol; j <= rightCol; j++) {
                System.out.println(matrix[topRow][j]);
            }
            topRow++;

            for (int i = topRow; i <= bottomRow; i++) {
                System.out.println(matrix[i][rightCol]);
            }
            rightCol--;

            if (topRow <= bottomRow) {
                for (int j = rightCol; j >= leftCol; j--) {
                    System.out.println(matrix[bottomRow][j]);
                }
                bottomRow--;
            }

            if (leftCol <= rightCol) {
                for (int i = bottomRow; i >= topRow; i--) {
                    System.out.println(matrix[i][leftCol]);
                }
                leftCol++;
            }
        }
    }

    // S18.
    public static int largestProductOfThree(int[] nums) {
        int n = nums.length;
        int largest1 = Integer.MIN_VALUE, largest2 = Integer.MIN_VALUE, largest3 = Integer.MIN_VALUE;
        int smallest1 = Integer.MAX_VALUE, smallest2 = Integer.MAX_VALUE;

        // Find the three largest and two smallest numbers in case two smallest numbers
        // are negative
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (num > largest1) {
                largest3 = largest2;
                largest2 = largest1;
                largest1 = num;
            } else if (num > largest2) {
                largest3 = largest2;
                largest2 = num;
            } else if (num > largest3) {
                largest3 = num;
            }

            if (num < smallest1) {
                smallest2 = smallest1;
                smallest1 = num;
            } else if (num < smallest2) {
                smallest2 = num;
            }
        }

        return Math.max(largest1 * largest2 * largest3, largest1 * smallest1 * smallest2);
    }

    // S19.
    public static int nthPerfectNumber(int n) {
        int count = 0;
        int num = 19;

        while (count < n) {
            int sum = 0;
            int temp = num;

            // Sum up the digits of the current number
            while (temp > 0) {
                sum += temp % 10;
                temp /= 10;
            }

            // If the sum is 10, increment the count
            if (sum == 10) {
                count++;
            }

            // If we've found the nth perfect number, return it
            if (count == n) {
                return num;
            }

            num += 9; // Otherwise, move on to the next number
        }

        return -1; // This line is unreachable, but required by Java
    }

    // S20.
    public static ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        ListNode next = null;

        while (current != null) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }

        return prev;
    }

    // S21.
    static class Interval {
        int start;
        int end;

        Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }

        public String toString() {
            return "(" + start + ", " + end + ")";
        }
    }

    class IntervalMerger {

        public static List<Interval> mergeIntervals(List<Interval> intervals) {
            Collections.sort(intervals, Comparator.comparingInt(interval -> interval.start));

            List<Interval> mergedIntervals = new ArrayList<>();
            Interval currentInterval = null;

            for (Interval interval : intervals) {
                if (currentInterval == null) {
                    // If this is the first interval, add it to the merged intervals list
                    currentInterval = interval;
                    mergedIntervals.add(currentInterval);
                } else if (interval.start <= currentInterval.end) {
                    // If the current interval overlaps with the previous interval, merge them
                    currentInterval.end = Math.max(currentInterval.end, interval.end);
                } else {
                    // If the current interval doesn't overlap with the previous interval, add it to
                    // the merged intervals list
                    currentInterval = interval;
                    mergedIntervals.add(currentInterval);
                }
            }

            return mergedIntervals;
        }
    }

    // S22.
    class DeepestNodeFinder {
        public static TreeNode<Character> findDeepestNode(TreeNode<Character> root) {
            if (root == null) {
                return null;
            }

            Queue<TreeNode<Character>> queue = new LinkedList<>();
            queue.offer(root);

            TreeNode<Character> deepestNode = null;

            while (!queue.isEmpty()) {
                int size = queue.size();

                for (int i = 0; i < size; i++) {
                    TreeNode<Character> node = queue.poll();

                    deepestNode = node;

                    if (node.left != null) {
                        queue.offer(node.left);
                    }

                    if (node.right != null) {
                        queue.offer(node.right);
                    }
                }
            }

            return deepestNode;
        }
    }

    // S23.
    class LetterCombinations {
        public static List<String> letterCombinations(String digits, Map<Character, char[]> digitToLetters) {
            List<String> result = new ArrayList<>();
            if (digits == null || digits.length() == 0) {
                return result;
            }
            backtrack(result, new StringBuilder(), digits, 0, digitToLetters);
            return result;
        }

        private static void backtrack(List<String> result, StringBuilder temp, String digits, int index,
                Map<Character, char[]> digitToLetters) {
            if (temp.length() == digits.length()) {
                result.add(temp.toString());
                return;
            }
            char[] letters = digitToLetters.get(digits.charAt(index));
            for (char letter : letters) {
                temp.append(letter);
                backtrack(result, temp, digits, index + 1, digitToLetters);
                temp.deleteCharAt(temp.length() - 1); // To reuse in next iteration
            }
        }
    }

    // S24.
    class FileReader {
        private int pos = 0;
        private String buffer = "";

        public String readN(int n) {
            StringBuilder sb = new StringBuilder();
            while (sb.length() < n) {
                if (buffer.length() == pos) { // need to read more characters from file
                    File file = new File("file.txt");
                    buffer = read7(file);
                    pos = 0;
                    if (buffer.isEmpty()) { // end of file reached
                        break;
                    }
                }
                sb.append(buffer.charAt(pos++));
            }
            return sb.toString();
        }

        private String read7(File file) {
            StringBuilder sb = new StringBuilder();
            try {
                FileInputStream fis = new FileInputStream(file);
                BufferedInputStream bis = new BufferedInputStream(fis);
                byte[] buffer = new byte[7];
                int bytesRead = bis.read(buffer);
                while (bytesRead != -1) {
                    sb.append(new String(buffer, 0, bytesRead));
                    bytesRead = bis.read(buffer);
                }
                bis.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return sb.toString();

        }
    }

    // S26.
    class MaxSumInBinaryTree {
        private int maxSum = Integer.MIN_VALUE;

        public int maxPathSum(TreeNode<Integer> root) {
            maxSumHelper(root);
            return maxSum;
        }

        private int maxSumHelper(TreeNode<Integer> node) {
            if (node == null) {
                return 0;
            }

            int leftSum = Math.max(maxSumHelper(node.left), 0);
            int rightSum = Math.max(maxSumHelper(node.right), 0);

            int currentSum = node.val + Math.max(leftSum, rightSum);
            maxSum = Math.max(maxSum, currentSum);

            return node.val + Math.max(leftSum, rightSum);
        }
    }

    // S27.
    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return result;
        }
        List<Integer> current = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        backtrack(nums, current, used, result);
        return result;
    }

    private static void backtrack(int[] nums, List<Integer> current, boolean[] used, List<List<Integer>> result) {
        if (current.size() == nums.length) {
            result.add(new ArrayList<>(current));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            current.add(nums[i]);
            used[i] = true;
            backtrack(nums, current, used, result);
            used[i] = false;
            current.remove(current.size() - 1);
        }
    }

    // S28.
    public static boolean exists(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0 || word == null) {
            return false;
        }
        int m = board.length;
        int n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (search(board, visited, word, i, j, 0)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static boolean search(char[][] board, boolean[][] visited, String word, int row, int col, int index) {
        if (index == word.length()) {
            return true;
        }
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length
                || visited[row][col] || board[row][col] != word.charAt(index)) {
            return false;
        }
        visited[row][col] = true;
        boolean found = search(board, visited, word, row - 1, col, index + 1)
                || search(board, visited, word, row + 1, col, index + 1)
                || search(board, visited, word, row, col - 1, index + 1)
                || search(board, visited, word, row, col + 1, index + 1);
        visited[row][col] = false;
        return found;
    }

    // S29.
    public static int minSteps(int[][] points) {
        int steps = 0;
        int[] current = points[0];
        for (int i = 1; i < points.length; i++) {
            int[] next = points[i];
            steps += Math.max(Math.abs(next[0] - current[0]), Math.abs(next[1] - current[1]));
            current = next;
        }
        return steps;
    }

    // S30.
    public static int[] getPrimes(int n) {
        boolean[] isPrime = new boolean[n + 1];
        Arrays.fill(isPrime, true);
        isPrime[0] = isPrime[1] = false;

        // Sieve of Eratosthenes
        for (int i = 2; i * i <= n; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j <= n; j += i) {
                    isPrime[j] = false;
                }
            }
        }

        int[] primes = new int[2];
        for (int i = 2; i <= n / 2; i++) {
            if (isPrime[i] && isPrime[n - i]) {
                primes[0] = i;
                primes[1] = n - i;
                break;
            }
        }
        return primes;
    }

    public static void main(String[] args) {
        /*
         * Q1.
         * Given a list of numbers and a number k, return whether any two numbers from
         * the list add up to k.
         * For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is
         * 17.
         * Bonus: Can you do this in one pass?
         */
        int[] nums = { 10, 15, 3, 7 };
        int k = 17;
        boolean result = doesTwoSumHaveK(nums, k);
        System.out.print("Do any two sums have k: ");
        System.out.print(result);

        /*
         * Q2.
         * A unival tree (which stands for "universal value") is a tree where all nodes
         * under it have the same value.
         * Given the root to a binary tree, count the number of unival subtrees.
         * For example, the following tree has 5 unival subtrees:
         * "     0       "
         * "    / \      "
         * "   1   0     "
         * "      / \    "
         * "     1   0   "
         * "    / \      "
         * "   1   1     "
         */
        TreeNode<Integer> root = new TreeNode<Integer>(0);
        root.left = new TreeNode<Integer>(1);
        root.right = new TreeNode<Integer>(0);
        root.right.left = new TreeNode<Integer>(1);
        root.right.right = new TreeNode<Integer>(0);
        root.right.left.left = new TreeNode<Integer>(1);
        root.right.left.right = new TreeNode<Integer>(1);

        UnivalTreeCount univalTreeCount = new UnivalTreeCount();
        System.out.print("Unival Tree Count: ");
        System.out.println(univalTreeCount.countUnivalSubtrees(root));

        /*
         * Q3.
         * Given two singly linked lists that intersect at some point, find the
         * intersecting node. The lists are non-cyclical.
         * For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10, return
         * the node with value 8.
         * In this example, assume nodes with the same value are the exact same node
         * objects.
         * Do this in O(M + N) time (where M and N are the lengths of the lists) and
         * constant space.
         */
        ListNode a = new ListNode(3);
        a.next = new ListNode(7);
        a.next.next = new ListNode(8);
        a.next.next.next = new ListNode(10);

        ListNode b = new ListNode(99);
        b.next = new ListNode(1);
        b.next.next = new ListNode(8);
        b.next.next.next = new ListNode(10);

        LinkedListIntersection linkedListIntersection = new LinkedListIntersection();
        ListNode intersection = linkedListIntersection.getIntersectionNode(a, b);
        System.out.print("Find intersecting node: ");
        System.out.println(intersection.val);

        /*
         * Q4.
         * Given an array of time intervals (start, end) for classroom lectures
         * (possibly overlapping), find the minimum number of rooms required.
         * For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.
         */
        int[][] intervals = { { 30, 75 }, { 0, 50 }, { 60, 150 } };

        MinimumRooms minimumRooms = new MinimumRooms();
        int numRooms = minimumRooms.minMeetingRooms(intervals);
        System.out.print("Minimum number of rooms: ");
        System.out.println(numRooms);

        /*
         * Q5.
         * You are given an M by N matrix consisting of booleans that represents a
         * board. Each True boolean represents a wall. Each False boolean represents a
         * tile you can walk on.
         * Given this matrix, a start coordinate, and an end coordinate, return the
         * minimum number of steps required to reach the end coordinate from the start.
         * If there is no possible path, then return null. You can move up, left, down,
         * and right. You cannot move through walls. You cannot wrap around the edges of
         * the board.
         * For example, given the following board:
         * [[f, f, f, f],
         * [t, t, f, t],
         * [f, f, f, f],
         * [f, f, f, f]]
         * and start = (3, 0) (bottom left) and end = (0, 0) (top left), the minimum
         * number of steps required to reach the end is 7, since we would need to go
         * through (1, 2) because there is a wall everywhere else on the second row.
         */
        boolean[][] board = { { false, false, false, false },
                { true, true, false, true },
                { false, false, false, false },
                { false, false, false, false } };
        Cell start = new Cell(3, 0);
        Cell end = new Cell(0, 0);
        MatrixPathFinder matrixPathFinder = new MatrixPathFinder();

        Integer steps = matrixPathFinder.findShortestPath(board, start, end);
        if (steps != null) {
            System.out.println("Minimum steps: " + steps);
        } else {
            System.out.println("No path found.");
        }

        /*
         * Q6.
         * You run an e-commerce website and want to record the last N order ids in a
         * log. Implement a data structure to accomplish this, with the following API:
         * record(order_id): adds the order_id to the log
         * get_last(i): gets the ith last element from the log. i is guaranteed to be
         * smaller than or equal to N.
         * You should be as efficient with time and space as possible.
         */

        // Solution implemented in S6 OrderLog class

        /*
         * Q7.
         * Given a string of round, curly, and square open and closing brackets, return
         * whether the brackets are balanced (well-formed).
         * For example, given the string "([])[]({})", you should return true.
         * Given the string "([)]" or "((()", you should return false.
         */
        String str1 = "([])[]({})";
        String str2 = "([)]";
        String str3 = "((()";

        BracketChecker bracketChecker = new BracketChecker();
        System.out.println(bracketChecker.isBalanced(str1));
        System.out.println(bracketChecker.isBalanced(str2));
        System.out.println(bracketChecker.isBalanced(str3));

        /*
         * Q8.
         * Run-length encoding is a fast and simple method of encoding strings. The
         * basic idea is to represent repeated successive characters as a single count
         * and character. For example, the string "AAAABBBCCDAA" would be encoded as
         * "4A3B2C1D2A".
         * Implement run-length encoding and decoding. You can assume the string to be
         * encoded have no digits and consists solely of alphabetic characters. You can
         * assume the string to be decoded is valid.
         */
        String input = "AAAABBBCCDAA";
        RunLengthEncoderDecoder runLengthEncoderDecoder = new RunLengthEncoderDecoder();
        String encoded = runLengthEncoderDecoder.encode(input);
        String decoded = runLengthEncoderDecoder.decode(encoded);
        System.out.println("Encoded string: " + encoded);
        System.out.println("Decoded string: " + decoded);

        /*
         * Q9.
         * The edit distance between two strings refers to the minimum number of
         * character insertions, deletions, and substitutions required to change one
         * string to the other. For example, the edit distance between “kitten” and
         * “sitting” is three: substitute the “k” for “s”, substitute the “e” for “i”,
         * and append a “g”.
         * Given two strings, compute the edit distance between them.
         */

        String s1 = "kitten";
        String s2 = "sitting";
        /*
         * Table for kitten and sitting
         * 0 1 2 3 4 5 6 7
         * 1 1 2 3 4 5 6 7
         * 2 2 1 2 3 4 5 6
         * 3 3 2 1 2 3 4 5
         * 4 4 3 2 1 2 3 4
         * 5 5 4 3 2 2 3 4
         * 6 6 5 4 3 3 2 3
         */
        int editDistanceOfTwoStrings = editDistance(s1, s2);
        System.out.print("Edit distance of two strings: ");
        System.out.println(editDistanceOfTwoStrings);

        /*
         * Q10.
         * Compute the running median of a sequence of numbers. That is, given a stream
         * of numbers, print out the median of the list so far on each new element.
         * Recall that the median of an even-numbered list is the average of the two
         * middle numbers.
         * For example, given the sequence [2, 1, 5, 7, 2, 0, 5], your algorithm should
         * print out:
         * 2
         * 1.5
         * 2
         * 3.5
         * 2
         * 2
         * 2
         */
        RunningMedian rm = new RunningMedian();
        rm.addNumber(2);
        System.out.println(rm.getMedian());
        rm.addNumber(1);
        System.out.println(rm.getMedian());
        rm.addNumber(5);
        System.out.println(rm.getMedian());
        rm.addNumber(7);
        System.out.println(rm.getMedian());
        rm.addNumber(2);
        System.out.println(rm.getMedian());
        rm.addNumber(0);
        System.out.println(rm.getMedian());
        rm.addNumber(5);
        System.out.println(rm.getMedian());

        /*
         * Q11.
         * The power set of a set is the set of all its subsets. Write a function that,
         * given a set, generates its power set.
         * For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3},
         * {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.
         * You may also use a list or array to represent a set.
         */
        Set<Integer> set = new HashSet<>(Arrays.asList(1, 2, 3));
        Set<Set<Integer>> powerSet = PowerSet.generatePowerSet(set);
        System.out.println(powerSet);

        /*
         * Q12.
         * Implement a stack that has the following methods:
         * push(val), which pushes an element onto the stack
         * pop(), which pops off and returns the topmost element of the stack. If there
         * are no elements in the stack, then it should throw an error or return null.
         * max(), which returns the maximum value in the stack currently. If there are
         * no elements in the stack, then it should throw an error or return null.
         * Each method should run in constant time.
         */
        MaxStack stack = new MaxStack();
        stack.push(3);
        stack.push(1);
        stack.push(5);
        System.out.println(stack.max()); // should print 5
        System.out.println(stack.pop()); // should print 5
        System.out.println(stack.max()); // should print 3

        /*
         * Q13.
         * Given a array of numbers representing the stock prices of a company in
         * chronological order, write a function that calculates the maximum profit you
         * could have made from buying and selling that stock once. You must buy before
         * you can sell it.
         * For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you could
         * buy the stock at 5 dollars and sell it at 10 dollars.
         */
        int[] stockHistory = { 9, 11, 8, 5, 7, 10 };
        int maxProfit = maxProfit(stockHistory);
        System.out.print("Max profit: ");
        System.out.println(maxProfit);

        /*
         * Q14.
         * Suppose an arithmetic expression is given as a binary tree. Each leaf is an
         * integer and each internal node is one of '+', '−', '∗', or '/'.
         * Given the root to such a tree, write a function to evaluate it.
         * For example, given the following tree:
         * "    *      "
         * "   / \     "
         * "  +    +   "
         * " / \  / \  "
         * "3  2  4  5 "
         * You should return 45, as it is (3 + 2) * (4 + 5).
         */
        TreeNode<Character> arithmeticExpTree = new TreeNode<Character>('*');
        arithmeticExpTree.left = new TreeNode<Character>('+');
        arithmeticExpTree.right = new TreeNode<Character>('+');
        arithmeticExpTree.left.left = new TreeNode<Character>('3');
        arithmeticExpTree.left.right = new TreeNode<Character>('2');
        arithmeticExpTree.right.left = new TreeNode<Character>('4');
        arithmeticExpTree.right.right = new TreeNode<Character>('5');
        int arithmeticExpTreeResult = evaluate(arithmeticExpTree);
        System.out.println(arithmeticExpTreeResult);

        /*
         * Q15.
         * Implement a URL shortener with the following methods:
         * shorten(url), which shortens the url into a six-character alphanumeric
         * string, such as zLg6wl.
         * restore(short), which expands the shortened string into the original url. If
         * no such shortened string exists, return null.
         * Hint: What if we enter the same URL twice?
         */
        String url = "user/create-order";
        UrlShortener urlShortener = new UrlShortener();
        String shortenedUrl = urlShortener.shorten(url);
        System.out.println(shortenedUrl);
        String restoredUrl = urlShortener.restore(shortenedUrl);
        System.out.println(restoredUrl);

        /*
         * Q16.
         * Given a 2D matrix of characters and a target word, write a function that
         * returns whether the word can be found in the matrix by going left-to-right,
         * or up-to-down.
         * For example, given the following matrix:
         * [['F', 'A', 'C', 'I'],
         * ['O', 'B', 'Q', 'P'],
         * ['A', 'N', 'O', 'B'],
         * ['M', 'A', 'S', 'S']]
         * and the target word 'FOAM', you should return true, since it's the leftmost
         * column. Similarly, given the target word 'MASS', you should return true,
         * since it's the last row.
         */
        char[][] matrix = { { 'F', 'A', 'C', 'I' }, { 'O', 'B', 'Q', 'P' }, { 'A', 'N', 'O', 'B' },
                { 'M', 'A', 'S', 'S' } };
        String target = "FOAM";
        boolean isPresent = isWordPresent(matrix, target);
        System.out.println(isPresent);

        target = "MASS";
        isPresent = isWordPresent(matrix, target);
        System.out.println(isPresent);

        /*
         * Q17.
         * Given a N by M matrix of numbers, print out the matrix in a clockwise spiral.
         * For example, given the following matrix:
         * [[1, 2, 3, 4, 5],
         * [6, 7, 8, 9, 10],
         * [11, 12, 13, 14, 15],
         * [16, 17, 18, 19, 20]]
         * You should print out the following:
         * 1
         * 2
         * 3
         * 4
         * 5
         * 10
         * 15
         * 20
         * 19
         * 18
         * 17
         * 16
         * 11
         * 6
         * 7
         * 8
         * 9
         * 14
         * 13
         * 12
         */
        int[][] matrixForSpiralPrint = { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 }, { 11, 12, 13, 14, 15 },
                { 16, 17, 18, 19, 20 } };
        printMatrixInSpiralOrder(matrixForSpiralPrint);

        /*
         * Q18.
         * Given a list of integers, return the largest product that can be made by
         * multiplying any three integers.
         * For example, if the list is [-10, -10, 5, 2], we should return 500, since
         * that's -10 * -10 * 5.
         * You can assume the list has at least three integers.
         */
        int[] listOfIntegers = { -10, -10, 5, 2 };
        int largestProduct = largestProductOfThree(listOfIntegers);
        System.out.println(largestProduct);

        /*
         * Q19.
         * A number is considered perfect if its digits sum up to exactly 10.
         * Given a positive integer n, return the n-th perfect number.
         * For example, given 1, you should return 19. Given 2, you should return 28.
         */
        int n = 1;
        int nthPerfectNumber = nthPerfectNumber(n);
        System.out.println(nthPerfectNumber);

        /*
         * Q20.
         * Given the head of a singly linked list, reverse it in-place.
         */
        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(3);

        ListNode reversed = reverseList(head);

        while (reversed != null) {
            System.out.print(reversed.val + " -> ");
            reversed = reversed.next;
        }
        System.out.print("null");

        /*
         * Q21.
         * Given a list of possibly overlapping intervals, return a new list of
         * intervals where all overlapping intervals have been merged.
         * The input list is not necessarily ordered in any way.
         * For example, given [(1, 3), (5, 8), (4, 10), (20, 25)], you should return
         * [(1, 3), (4, 10), (20, 25)].
         */
        List<Interval> listOfIntervals = new ArrayList<>();
        listOfIntervals.add(new Interval(1, 3));
        listOfIntervals.add(new Interval(5, 8));
        listOfIntervals.add(new Interval(4, 10));
        listOfIntervals.add(new Interval(20, 25));

        List<Interval> mergedIntervals = IntervalMerger.mergeIntervals(listOfIntervals);
        System.out.println("Merged intervals: " + mergedIntervals);

        /*
         * Q22.
         * Given the root of a binary tree, return a deepest node. For example, in the
         * following tree, return d.
         * "    a      "
         * "   / \     "
         * "  b   c    "
         * " /         "
         * "d          "
         */
        TreeNode<Character> binaryRoot = new TreeNode<Character>('a');
        binaryRoot.left = new TreeNode<Character>('b');
        binaryRoot.right = new TreeNode<Character>('c');
        binaryRoot.left.left = new TreeNode<Character>('d');

        TreeNode<Character> deepestNode = DeepestNodeFinder.findDeepestNode(binaryRoot);
        System.out.println("Deepest node value: " + deepestNode.val);

        /*
         * Q23.
         * Given a mapping of digits to letters (as in a phone number), and a digit
         * string, return all possible letters the number could represent. You can
         * assume each valid number in the mapping is a single digit.
         * For example if {“2”: [“a”, “b”, “c”], 3: [“d”, “e”, “f”], …} then “23” should
         * return [“ad”, “ae”, “af”, “bd”, “be”, “bf”, “cd”, “ce”, “cf"].
         */
        Map<Character, char[]> digitToLetters = new HashMap<>();
        digitToLetters.put('2', new char[] { 'a', 'b', 'c' });
        digitToLetters.put('3', new char[] { 'd', 'e', 'f' });
        digitToLetters.put('4', new char[] { 'g', 'h', 'i' });
        digitToLetters.put('5', new char[] { 'j', 'k', 'l' });
        digitToLetters.put('6', new char[] { 'm', 'n', 'o' });
        digitToLetters.put('7', new char[] { 'p', 'q', 'r', 's' });
        digitToLetters.put('8', new char[] { 't', 'u', 'v' });
        digitToLetters.put('9', new char[] { 'w', 'x', 'y', 'z' });

        String digits = "23";
        List<String> lettersForDigit = LetterCombinations.letterCombinations(digits, digitToLetters);

        System.out.println("Letter combinations for " + digits + ": " + lettersForDigit);

        /*
         * Q24.
         * Using a read7() method that returns 7 characters from a file, implement
         * readN(n) which reads n characters.
         * For example, given a file with the content “Hello world”, three read7()
         * returns “Hello w”, “orld” and then “”.
         */
        // Solution implemented in S24 (NOT tested)

        /*
         * Q25.
         * What does the below code snippet print out? How can we fix the anonymous
         * functions to behave as we'd expect?
         * 
         * functions = []
         * for i in range(10):
         * functions.append(lambda : i)
         * 
         * for f in functions:
         * print(f())
         */
        // It will print '9' tem times as the lambda functions created using `lambda: i`
        // all reference the same variable 'i'
        // CORRECTION: functions.append(lambda x=i: x)

        /*
         * Q26.
         * Given a binary tree of integers, find the maximum path sum between two nodes.
         * The path must go through at least one node, and does not need to go through
         * the root.
         */
        // Solution implemented in S26 (NOT tested)

        /*
         * Q27.
         * Given a number in the form of a list of digits, return all possible
         * permutations.
         * For example, given [1,2,3], return
         * [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]].
         */
        int[] list = { 1, 2, 3 };
        List<List<Integer>> permutations = permute(list);
        System.out.print("Permutations for [1, 2, 3]: ");
        System.out.println(permutations);

        /*
         * Q28.
         * Given a 2D board of characters and a word, find if the word exists in the
         * grid.
         * The word can be constructed from letters of sequentially adjacent cell, where
         * "adjacent" cells are those horizontally or vertically neighboring. The same
         * letter cell may not be used more than once.
         * For example, given the following board:
         * [
         * ['A','B','C','E'],
         * ['S','F','C','S'],
         * ['A','D','E','E']
         * ]
         * exists(board, "ABCCED") returns true, exists(board, "SEE") returns true,
         * exists(board, "ABCB") returns false.
         */
        char[][] characterBoard = { { 'A', 'B', 'C', 'D' }, { 'S', 'F', 'C', 'S' }, { 'A', 'D', 'E', 'E' } };
        String word1 = "ABCCED";
        String word2 = "SEE";
        String word3 = "ABCB";
        boolean exists = exists(characterBoard, word1);
        System.out.println(exists);
        System.out.println(exists(characterBoard, word2));
        System.out.println(exists(characterBoard, word3));

        /*
         * Q29.
         * You are in an infinite 2D grid where you can move in any of the 8 directions:
         * (x,y) to
         * (x+1, y),
         * (x - 1, y),
         * (x, y+1),
         * (x, y-1),
         * (x-1, y-1),
         * (x+1,y+1),
         * (x-1,y+1),
         * (x+1,y-1)
         * You are given a sequence of points and the order in which you need to cover
         * the points. Give the minimum number of steps in which you can achieve it. You
         * start from the first point.
         * Example:
         * Input: [(0, 0), (1, 1), (1, 2)]
         * Output: 2
         * It takes 1 step to move from (0, 0) to (1, 1). It takes one more step to move
         * from (1, 1) to (1, 2).
         */
        int[][] points = { { 0, 0 }, { 1, 1 }, { 1, 2 } };
        int minSteps = minSteps(points);
        System.out.println(minSteps);

        /*
         * Q30.
         * Given an even number (greater than 2), return two prime numbers whose sum
         * will be equal to the given number.
         * A solution will always exist. See Goldbach’s conjecture.
         * https://en.wikipedia.org/wiki/Goldbach%27s_conjecture
         * Example:
         * Input: 4
         * Output: 2 + 2 = 4
         * If there are more than one solution possible, return the lexicographically
         * smaller solution.
         * If [a, b] is one solution with a <= b, and [c, d] is another solution with c
         * <= d, then
         * [a, b] < [c, d]
         * If a < c OR a==c AND b < d.
         */
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an even number greater than 2: ");
        int num = sc.nextInt();
        int[] primes = getPrimes(num);
        System.out.printf("%d + %d = %d\n", primes[0], primes[1], num);
        sc.close();
    }
}
