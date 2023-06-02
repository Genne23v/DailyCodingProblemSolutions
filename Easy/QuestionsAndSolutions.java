import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Set;
import java.util.HashSet;
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
import java.util.concurrent.TimeUnit;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.math.BigInteger;

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
    class FileByteReader {
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

    // S31.
    static class DoublyLinkedListNode {
        int data;
        DoublyLinkedListNode next;
        DoublyLinkedListNode prev;

        DoublyLinkedListNode(int data) {
            this.data = data;
            this.next = null;
            this.prev = null;
        }
    }

    static class DoublyLinkedList {
        DoublyLinkedListNode head;
        DoublyLinkedListNode tail;

        DoublyLinkedList() {
            this.head = null;
            this.tail = null;
        }

        void add(int data) {
            DoublyLinkedListNode node = new DoublyLinkedListNode(data);
            if (head == null) {
                head = node;
                tail = node;
            } else {
                tail.next = node;
                node.prev = tail;
                tail = node;
            }
        }

        boolean isPalindrome() {
            DoublyLinkedListNode start = head;
            DoublyLinkedListNode end = tail;

            while (start != null && end != null) {
                if (start.data != end.data) {
                    return false;
                }
                start = start.next;
                end = end.prev;
            }
            return true;
        }
    }

    static class SinglyLinkedListNode {
        int data;
        SinglyLinkedListNode next;

        SinglyLinkedListNode(int data) {
            this.data = data;
            this.next = null;
        }
    }

    static class SinglyLinkedList {
        SinglyLinkedListNode head;

        SinglyLinkedList() {
            this.head = null;
        }

        void add(int data) {
            SinglyLinkedListNode node = new SinglyLinkedListNode(data);
            if (head == null) {
                head = node;
            } else {
                SinglyLinkedListNode temp = head;
                while (temp.next != null) {
                    temp = temp.next;
                }
                temp.next = node;
            }
        }

        boolean isPalindrome() {
            if (head == null || head.next == null) {
                return true;
            }

            SinglyLinkedListNode slow = head;
            SinglyLinkedListNode fast = head;

            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
            }

            SinglyLinkedListNode secondHalf = reverse(slow);

            SinglyLinkedListNode temp1 = head;
            SinglyLinkedListNode temp2 = secondHalf;

            while (temp2 != null) {
                if (temp1.data != temp2.data) {
                    reverse(secondHalf);
                    return false;
                }
                temp1 = temp1.next;
                temp2 = temp2.next;
            }
            reverse(secondHalf);
            return true;
        }

        SinglyLinkedListNode reverse(SinglyLinkedListNode head) {
            SinglyLinkedListNode prev = null;
            SinglyLinkedListNode curr = head;
            SinglyLinkedListNode next = null;

            while (curr != null) {
                next = curr.next;
                curr.next = prev;
                prev = curr;
                curr = next;
            }
            head = prev;

            return head;
        }
    }

    // S32.
    static class Debounce {
        private Runnable runnable;
        private long delay;
        private long lastExecutionTime;

        public Debounce(Runnable runnable, long delay) {
            this.runnable = runnable;
            this.delay = delay;
        }

        public void execute() {
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastExecutionTime > delay) {
                lastExecutionTime = currentTime;
                runnable.run(); // Run when delay amount of time has passed
            }
        }
    }

    // S33.
    public static <T> void printLevelOrder(TreeNode<T> root) {
        if (root == null) {
            return;
        }

        Queue<TreeNode<T>> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            TreeNode<T> node = queue.remove();
            System.out.print(node.val + " ");

            if (node.left != null) {
                queue.add(node.left);
            }

            if (node.right != null) {
                queue.add(node.right);
            }
        }
    }

    // S34.
    public static boolean canShift(String A, String B) {
        if (A.length() != B.length()) {
            return false;
        }
        String A2 = A + A;
        return A2.contains(B);
    }

    // S35.
    public static int minLevelSum(TreeNode<Integer> root) {
        if (root == null) {
            return -1;
        }

        Queue<TreeNode<Integer>> queue = new LinkedList<>();
        queue.offer(root);
        int minLevel = 0;
        int minSum = Integer.MAX_VALUE;
        int level = 0;

        while (!queue.isEmpty()) {
            int size = queue.size();
            int sum = 0;

            for (int i = 0; i < size; i++) {
                TreeNode<Integer> node = queue.poll();
                sum += node.val;

                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }

            if (sum < minSum) {
                minSum = sum;
                minLevel = level;
            }

            level++;
        }

        return minLevel;
    }

    // S36.
    public static int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        int left = 0, right = n - 1;

        for (int i = n - 1; i >= 0; i--) {
            if (Math.abs(nums[left]) > Math.abs(nums[right])) {
                result[i] = nums[left] * nums[left];
                left++;
            } else {
                result[i] = nums[right] * nums[right];
                right--;
            }
        }

        return result;
    }

    // S37.
    public static int numRounds(int n) {
        if (n == 1) {
            return 0;
        } else {
            return 1 + numRounds(n / 2);
        }
    }

    // S38.
    public static TreeNode<Integer>[] findTarget(TreeNode<Integer> root, int k) {
        List<Integer> list = new ArrayList<Integer>();
        inorder(root, list);
        int left = 0, right = list.size() - 1;
        while (left < right) {
            int sum = list.get(left) + list.get(right);
            if (sum == k) {
                TreeNode<Integer>[] result = new TreeNode[2];
                result[0] = findNode(root, list.get(left));
                result[1] = findNode(root, list.get(right));
                return result;
            } else if (sum < k) {
                left++;
            } else {
                right--;
            }
        }
        return null;
    }

    private static void inorder(TreeNode<Integer> root, List<Integer> list) {
        if (root == null) {
            return;
        }
        inorder(root.left, list);
        list.add(root.val);
        inorder(root.right, list);
    }

    private static TreeNode<Integer> findNode(TreeNode<Integer> root, int val) {
        if (root == null || root.val == val) {
            return root;
        }
        if (val < root.val) {
            return findNode(root.left, val);
        } else {
            return findNode(root.right, val);
        }
    }

    // S39.
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        int carry = 0;

        while (l1 != null || l2 != null) {
            int sum = carry;

            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }

            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }

            current.next = new ListNode(sum % 10);
            current = current.next;
            carry = sum / 10;
        }

        if (carry > 0) {
            current.next = new ListNode(carry);
        }

        return dummy.next;
    }

    // S40.
    static class HitCounter {
        private List<Long> timestamps;

        public HitCounter() {
            timestamps = new ArrayList<>();
        }

        public void record(long timestamp) {
            timestamps.add(timestamp);
        }

        public int total() {
            return timestamps.size();
        }

        public int range(long lower, long upper) {
            int count = 0;
            for (int i = 0; i < timestamps.size(); i++) {
                if (timestamps.get(i) >= lower && timestamps.get(i) <= upper) {
                    count++;
                }
            }
            return count;
        }
    }

    // S41.
    static class SparseArray {
        private HashMap<Integer, Integer> map;
        private int size;

        public SparseArray(int[] arr, int size) {
            this.map = new HashMap<>();
            this.size = size;
            for (int i = 0; i < arr.length; i++) {
                if (arr[i] != 0) {
                    map.put(i, arr[i]);
                }
            }
        }

        public void set(int i, int val) {
            if (i < 0 || i >= size) {
                throw new IndexOutOfBoundsException();
            }
            if (val == 0) {
                map.remove(i);
            } else {
                map.put(i, val);
            }
        }

        public int get(int i) {
            if (i < 0 || i >= size) {
                throw new IndexOutOfBoundsException();
            }
            return map.getOrDefault(i, 0);
        }
    }

    // S42.
    static class MinimumPathSum {
        public int minPathSum(TreeNode<Integer> root) {
            if (root == null) {
                return 0;
            }
            return findMinPath(root, 0);
        }

        private int findMinPath(TreeNode<Integer> node, int currentSum) {
            if (node == null) {
                return Integer.MAX_VALUE;
            }

            currentSum += node.val;

            if (node.left == null && node.right == null) {
                return currentSum;
            }

            int leftSum = findMinPath(node.left, currentSum);
            int rightSum = findMinPath(node.right, currentSum);

            return Math.min(leftSum, rightSum);
        }
    }

    // S43.
    static class SwapNodesInPairs {
        public ListNode swapPairs(ListNode head) {
            ListNode dummy = new ListNode(0);
            dummy.next = head;
            ListNode prev = dummy;

            while (head != null && head.next != null) {
                ListNode first = head;
                ListNode second = head.next;

                prev.next = second;
                first.next = second.next;
                second.next = first;

                prev = first;
                head = first.next;
            }

            return dummy.next;
        }
    }

    // S44.
    static class StackUsingHeap {
        private int priority;
        private PriorityQueue<HeapNode> heap;

        public StackUsingHeap() {
            this.priority = 0;
            this.heap = new PriorityQueue<>((a, b) -> b.priority - a.priority);
        }

        public void push(int item) {
            heap.offer(new HeapNode(item, priority++));
        }

        public int pop() {
            if (heap.isEmpty()) {
                throw new IllegalStateException("Stack is empty");
            }

            return heap.poll().value;
        }

        private static class HeapNode {
            private int value;
            private int priority;

            HeapNode(int value, int priority) {
                this.value = value;
                this.priority = priority;
            }
        }
    }

    // S45.
    public static boolean isPermutationPalindrome(String str) {
        Map<Character, Integer> charFrequency = new HashMap<>();

        for (char c : str.toCharArray()) {
            charFrequency.put(c, charFrequency.getOrDefault(c, 0) + 1);
        }

        int oddCount = 0;

        for (int count : charFrequency.values()) {
            if (count % 2 != 0) {
                oddCount++;
            }
        }

        return oddCount <= 1;
    }

    // S46.
    public static Character findFirstRecurringCharacter(String str) {
        Set<Character> seenChars = new HashSet<>();

        for (char c : str.toCharArray()) {
            if (seenChars.contains(c)) {
                return c;
            }
            seenChars.add(c);
        }
        return null;
    }

    // S47.
    public static String reverseBinary(String binary) {
        long number = Long.parseLong(binary, 2);

        Long reversed = 0L;

        for (int i = 0; i < 32; i++) {
            reversed <<= 1; // Shift the reversed bits to the left
            reversed |= (number & 1); // Set the least significant bit of reversed to the current bit of number
            number >>= 1; // Shift the bits of number to the right
        }

        return Long.toBinaryString(reversed); // Convert reversed integer to binary string
    }

    // S48.
    public static int[] findBusiestPeriod(int[][] data) {
        HashMap<Integer, Integer> countMap = new HashMap<>();
        int maxCount = 0;
        int currentCount = 0;
        int startTimestamp = 0;
        int endTimestamp = 0;
        boolean maxed = false;

        for (int[] entry : data) {
            int timestamp = entry[0];
            int count = entry[1];
            String type = entry[2] == 1 ? "enter" : "exit";

            if (type.equals("enter")) {
                currentCount += count;
            } else {
                currentCount -= count;
            }

            countMap.put(timestamp, currentCount);

            if (currentCount > maxCount) {
                maxed = true;
                maxCount = currentCount;
                startTimestamp = timestamp;
                endTimestamp = timestamp;
            } else if (maxed && currentCount == maxCount) {
                endTimestamp = timestamp;
                maxed = false;
            } else if (maxed && currentCount < maxCount) {
                endTimestamp = timestamp - 1;
                maxed = false;
            }
        }

        return new int[] { startTimestamp, endTimestamp };
    }

    // S49.
    public static Map<String, Object> flattenDictionary(Map<String, Object> dict) {
        Map<String, Object> flattenedDict = new HashMap<>();
        flattenDictionaryHelper("", dict, flattenedDict);
        return flattenedDict;
    }

    private static void flattenDictionaryHelper(String prefix, Map<String, Object> dict,
            Map<String, Object> flattenedDict) {
        for (Map.Entry<String, Object> entry : dict.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            if (value instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> nestedDict = (Map<String, Object>) value;
                flattenDictionaryHelper(prefix + key + ".", nestedDict, flattenedDict);
            } else {
                flattenedDict.put(prefix + key, value);
            }
        }
    }

    // S50.
    public static Map<Character, Integer> simulateMarkovChain(char start, int numSteps,
            List<TransitionProbability> transitionProbabilities) {
        Map<Character, Integer> stateCounts = new HashMap<>();
        stateCounts.put(start, 1);

        Random random = new Random();

        char currentState = start;
        for (int i = 0; i < numSteps; i++) {
            double randomValue = random.nextDouble();
            double cumulativeProbability = 0.0;

            for (TransitionProbability transition : transitionProbabilities) {
                if (transition.fromState == currentState) {
                    cumulativeProbability += transition.probability;

                    if (randomValue <= cumulativeProbability) {
                        currentState = transition.toState;
                        stateCounts.put(currentState, stateCounts.getOrDefault(currentState, 0) + 1);
                        break;
                    }
                }
            }
        }

        return stateCounts;
    }

    static class TransitionProbability {
        char fromState;
        char toState;
        double probability;

        public TransitionProbability(char fromState, char toState, double probability) {
            this.fromState = fromState;
            this.toState = toState;
            this.probability = probability;
        }
    }

    // S51.
    public static boolean isCharacterMapping(String s1, String s2) {
        if (s1.length() != s2.length()) { // Lengths should be equal for one-to-one mapping
            return false;
        }

        Map<Character, Character> mapping = new HashMap<>();

        for (int i = 0; i < s1.length(); i++) {
            char c1 = s1.charAt(i);
            char c2 = s2.charAt(i);

            if (mapping.containsKey(c1)) {
                if (mapping.get(c1) != c2) {
                    return false; // Different mapping already exists for c1
                }
            } else {
                mapping.put(c1, c2); // Add new mapping
            }
        }

        return true;
    }

    // S52.
    public static ListNode rotateRight(ListNode head, int k) {
        if (head == null || k == 0) {
            return head;
        }

        int length = getLength(head);
        k %= length;

        if (k == 0) {
            return head;
        }

        ListNode current = head;
        for (int i = 0; i < length - k - 1; i++) {
            current = current.next;
        }

        ListNode newHead = current.next;
        current.next = null;

        ListNode temp = newHead;
        while (temp.next != null) {
            temp = temp.next;
        }

        temp.next = head;

        return newHead;
    }

    private static int getLength(ListNode head) {
        int length = 0;
        ListNode current = head;
        while (current != null) {
            length++;
            current = current.next;
        }
        return length;
    }

    // S53.
    public static BigInteger greatestCommonDenominator(int[] nums) {
        BigInteger gcd = BigInteger.valueOf(nums[0]);

        for (int i = 1; i < nums.length; i++) {
            gcd = gcd(gcd, BigInteger.valueOf(nums[i]));
        }

        return gcd;
    }

    private static BigInteger gcd(BigInteger a, BigInteger b) {
        return a.gcd(b);
    }

    // S54.
    static class Rectangle {
        int topLeftX;
        int topLeftY;
        int width;
        int height;

        public Rectangle(int topLeftX, int topLeftY, int width, int height) {
            this.topLeftX = topLeftX;
            this.topLeftY = topLeftY;
            this.width = width;
            this.height = height;
        }

        public int getIntersectionArea(Rectangle other) {
            int left = Math.max(this.topLeftX, other.topLeftX);
            int right = Math.min(this.topLeftX + this.width, other.topLeftX + other.width);
            int top = Math.max(this.topLeftX, other.topLeftX);
            int bottom = Math.min(this.topLeftY + this.height, other.topLeftY + other.height);

            if (left > right || top > bottom) {
                return 0;
            }

            int intersectionWidth = right - left;
            int intersectionHeight = bottom - top;

            return intersectionWidth * intersectionHeight;
        }

        // S55.
        public boolean overlaps(Rectangle other) {
            int thisRight = this.topLeftX + this.width;
            int thisBottom = this.topLeftY + this.height;
            int otherRight = other.topLeftX + other.width;
            int otherBottom = other.topLeftY + other.height;

            return this.topLeftX < otherRight &&
                    thisRight > other.topLeftX &&
                    this.topLeftY < otherBottom &&
                    thisBottom > other.topLeftY;
        }
    }

    // S56.
    public static int findLongestSubarrayLength(int[] nums) {
        int n = nums.length;
        int maxLength = 0;
        int left = 0;
        int right = 0;
        HashSet<Integer> distinctSet = new HashSet<>();

        while (right < n) {
            if (!distinctSet.contains(nums[right])) {
                distinctSet.add(nums[right]);
                maxLength = Math.max(maxLength, right - left + 1);
                right++;
            } else {
                distinctSet.remove(nums[left]);
                left++;
            }
        }

        return maxLength;
    }

    // S57.
    public static int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) {
            return 0;
        }

        // Sort the intervals based on their end time in ascending order
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[1]));

        int nonOverlapCount = 1;
        int end = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {
            // If the start time of the current interval is after the end time of the
            // previous non-overlapping interval, it's non-overlapping
            if (intervals[i][0] >= end) {
                nonOverlapCount++;
                end = intervals[i][1];
            }
        }

        int minIntervalsToRemove = intervals.length - nonOverlapCount;

        return minIntervalsToRemove;
    }

    // S58.
    static class Segment {
        int start;
        int end;

        public Segment(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    public static int countIntersectingPairs(int[] p, int[] q) {
        int count = 0;
        int n = p.length;
        List<Segment> segments = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            Segment segment = new Segment(p[i], q[i]);
            segments.add(segment);
        }

        Collections.sort(segments, Comparator.comparingInt(segment -> segment.start));

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (segments.get(i).end > segments.get(j).end) {
                    count++;
                }
            }
        }

        return count;
    }

    // S59.
    public static int findMostFrequentSubtreeSum(TreeNode<Integer> root) {
        if (root == null) {
            return 0;
        }

        Map<Integer, Integer> sumFrequencies = new HashMap<>();
        calculateSubtreeSum(root, sumFrequencies);

        int maxFrequency = 0;
        int mostFrequentSum = 0;

        for (Map.Entry<Integer, Integer> entry : sumFrequencies.entrySet()) {
            int sum = entry.getKey();
            int frequency = entry.getValue();

            if (frequency > maxFrequency) {
                maxFrequency = frequency;
                mostFrequentSum = sum;
            }
        }

        return mostFrequentSum;
    }

    private static int calculateSubtreeSum(TreeNode<Integer> node, Map<Integer, Integer> sumFrequencies) {
        if (node == null) {
            return 0;
        }

        int leftSum = calculateSubtreeSum(node.left, sumFrequencies);
        int rightSum = calculateSubtreeSum(node.right, sumFrequencies);

        int currentSum = node.val + leftSum + rightSum;
        sumFrequencies.put(currentSum, sumFrequencies.getOrDefault(currentSum, 0) + 1);

        return currentSum;
    }

    // S60.
    public static void rotate(int[] nums, int k) {
        int n = nums.length;
        k = k % n;

        reverse(nums, 0, n - 1);
        // Reverse the first k elements
        reverse(nums, 0, k - 1);
        // Reverse the remaining elements
        reverse(nums, k, n - 1);
    }

    private static void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    // S61.
    public static int maximumPathSum(int[][] triangle) {
        int rows = triangle.length;

        int[][] memo = new int[rows][rows];

        // Initialize the bottom row of the memoization array with the values from the
        // triangle
        for (int i = 0; i < rows; i++) {
            memo[rows - 1][i] = triangle[rows - 1][i];
        }

        // Calculate the maximum path sum for each row from bottom to top
        for (int i = rows - 2; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                // Compute the maximum path sum by choosing the larger adjacent value below
                memo[i][j] = triangle[i][j] + Math.max(memo[i + 1][j], memo[i + 1][j + 1]);
            }
        }

        return memo[0][0];
    }

    // S62.
    public static boolean isPalindrome(int number) {
        if (number < 0 || (number != 0 && number % 10 == 0)) {
            return false;
        }

        int reversed = 0;
        int original = number;

        while (number != 0) {
            int digit = number % 10;
            reversed = reversed * 10 + digit;
            number /= 10;
        }

        return original == reversed;
    }

    // S63.
    public static boolean isCompleteBinaryTree(TreeNode<Integer> root) {
        int nodeCount = countNodes(root);
        int height = getHeight(root);
        int maxNodeCount = (1 << height) - 1;

        return nodeCount == maxNodeCount;
    }

    private static int countNodes(TreeNode<Integer> root) {
        if (root == null) {
            return 0;
        }

        int leftHeight = getLeftHeight(root);
        int rightHeight = getRightHeight(root);

        if (leftHeight == rightHeight) {
            // Last level is fully filled
            return (1 << leftHeight) - 1;
        } else {
            // Last level is partially filled
            return countNodes(root.left) + countNodes(root.right) + 1;
        }
    }

    private static int getLeftHeight(TreeNode<Integer> node) {
        int height = 0;
        while (node != null) {
            height++;
            node = node.left;
        }
        return height;
    }

    private static int getRightHeight(TreeNode<Integer> node) {
        int height = 0;
        while (node != null) {
            height++;
            node = node.right;
        }
        return height;
    }

    private static int getHeight(TreeNode<Integer> node) {
        return getLeftHeight(node);
    }

    // S64.
    public static int findNextPermutation(int num) {
        int[] digits = convertToDigits(num);

        // Find the first decreasing digit from right to left
        int pivotIndex = findPivotIndex(digits);

        // If no pivot is found, return the integer itself
        if (pivotIndex == -1) {
            return num;
        }

        // Find the smallest digit greater than pivot to the right of pivot
        int swapIndex = findSwapIndex(digits, pivotIndex);

        // Swap pivot and swap
        swap(digits, pivotIndex, swapIndex);

        // Reverse the digits to the right of pivot
        reverseNumber(digits, pivotIndex + 1, digits.length - 1);

        return convertToInt(digits);
    }

    private static int[] convertToDigits(int num) {
        String numStr = Integer.toString(num);
        int[] digits = new int[numStr.length()];
        for (int i = 0; i < numStr.length(); i++) {
            digits[i] = numStr.charAt(i) - '0';
        }
        return digits;
    }

    private static int findPivotIndex(int[] digits) {
        for (int i = digits.length - 2; i >= 0; i--) {
            if (digits[i] < digits[i + 1]) {
                return i;
            }
        }
        return -1;
    }

    private static int findSwapIndex(int[] digits, int pivotIndex) {
        int pivot = digits[pivotIndex];
        for (int i = digits.length - 1; i > pivotIndex; i--) {
            if (digits[i] > pivot) {
                return i;
            }
        }
        return -1;
    }

    private static void swap(int[] digits, int i, int j) {
        int temp = digits[i];
        digits[i] = digits[j];
        digits[j] = temp;
    }

    private static void reverseNumber(int[] digits, int start, int end) {
        while (start < end) {
            swap(digits, start, end);
            start++;
            end--;
        }
    }

    private static int convertToInt(int[] digits) {
        int result = 0;
        for (int digit : digits) {
            result = result * 10 + digit;
        }
        return result;
    }

    // S65.
    public static <T> T[] applyPermutation(T[] array, int[] permutation) {
        if (array.length != permutation.length) {
            throw new IllegalArgumentException("Array and permutation must have the same length.");
        }

        T[] result = array.clone();

        for (int i = 0; i < permutation.length; i++) {
            result[i] = array[permutation[i]];
        }

        return result;
    }

    // S66.
    public static long collatzSequence(long n) {
        if (n <= 0) {
            System.out.println(n);
            throw new IllegalArgumentException("Input must be a positive integer.");
        }

        if (n == 1) {
            return 0;
        }

        if (n % 2 == 0) {
            return 1 + collatzSequence(n / 2);
        } else {
            return 1 + collatzSequence(3 * n + 1);
        }
    }

    // S67.
    public static String getColumnID(int columnNumber) {
        StringBuilder columnID = new StringBuilder();

        while (columnNumber > 0) {
            int remainder = (columnNumber - 1) % 26;
            char c = (char) ('A' + remainder);
            columnID.insert(0, c);

            columnNumber = (columnNumber - 1) / 26;
        }

        return columnID.toString();
    }

    // S68.
    public static int longestConsecutiveRun(int n) {
        String binary = Integer.toBinaryString(n);
        int maxLength = 0;
        int currentLength = 0;

        for (char c : binary.toCharArray()) {
            if (c == '1') {
                currentLength++;
                maxLength = Math.max(maxLength, currentLength);
            } else {
                currentLength = 0;
            }
        }

        return maxLength;
    }

    // S69.
    public static int getNthSevenishNumber(int n) {
        int[] sevenishNumbers = new int[n];
        sevenishNumbers[0] = 1;
        int nextPowerOf7Index = 1;
        int powerIndex = 1;
        int nextIndexForSum = 1;
        int currentPowerOf7 = 7;
        int nextPowerOf7 = 7;

        for (int i = 1; i < n; i++) {
            if (i == nextPowerOf7Index) {
                sevenishNumbers[i] = nextPowerOf7;

                nextPowerOf7Index += (int) Math.pow(2, powerIndex);
                powerIndex++;

                currentPowerOf7 = nextPowerOf7;
                nextPowerOf7 *= 7;
                nextIndexForSum = 0;
            } else {
                sevenishNumbers[i] = currentPowerOf7 + sevenishNumbers[nextIndexForSum];
                nextIndexForSum++;
            }
        }

        return sevenishNumbers[n - 1];
    }

    // S70.
    public static int findSmallestPositiveInteger(int[] nums) {
        int smallestInteger = 1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= smallestInteger) {
                // If the current number in the array is less than or equal to the smallest
                // integer,
                // update the smallest integer by adding the current number
                smallestInteger += nums[i];
            } else {
                // If the current number in the array is greater than the smallest integer,
                // we have found the smallest positive integer that cannot be formed by the
                // subset sum
                break;
            }
        }
        return smallestInteger;
    }

    // S71.
    public static int findLastPrisoner(int N, int k) {
        if (k == 2) {
            // O(log N) solution for k = 2
            int highestPowerOf2 = highestPowerOf2(N);
            return 2 * (N - highestPowerOf2) + 1;
        }

        List<Integer> prisoners = new ArrayList<>();
        for (int i = 1; i <= N; i++) {
            prisoners.add(i);
        }

        int index = 0;
        while (prisoners.size() > 1) {
            // Compute the next index to remove
            index = (index + k - 1) % prisoners.size();

            // Remove the prisoner at the computed index
            prisoners.remove(index);
        }

        return prisoners.get(0);
    }

    private static int highestPowerOf2(int N) {
        int powerOf2 = 1;
        while (powerOf2 * 2 <= N) {
            powerOf2 *= 2;
        }
        return powerOf2;
    }

    // S72.
    static class TrieNode {
        private static final int ALPHABET_SIZE = 26;
        TrieNode[] children;
        boolean isEndOfWord;

        TrieNode() {
            children = new TrieNode[ALPHABET_SIZE];
            isEndOfWord = false;
        }
    }

    static class Trie {
        private TrieNode root;

        Trie() {
            root = new TrieNode();
        }

        void insert(String word) {
            TrieNode current = root;
            for (int i = 0; i < word.length(); i++) {
                int index = word.charAt(i) - 'A';
                if (current.children[index] == null) {
                    current.children[index] = new TrieNode();
                }
                current = current.children[index];
            }
            current.isEndOfWord = true;
        }

        boolean search(String word) {
            TrieNode current = root;
            for (int i = 0; i < word.length(); i++) {
                int index = word.charAt(i) - 'A';
                if (current.children[index] == null) {
                    return false;
                }
                current = current.children[index];
            }
            return current != null && current.isEndOfWord;
        }
    }

    public static Set<String> findWords(char[][] board, String[] dictionary) {
        Trie trie = new Trie();
        for (String word : dictionary) {
            trie.insert(word);
        }

        Set<String> foundWords = new HashSet<>();
        int rows = board.length;
        int columns = board[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                boolean[][] visited = new boolean[rows][columns];
                StringBuilder currentWord = new StringBuilder();
                dfs(board, visited, i, j, currentWord, trie, foundWords);
            }
        }

        return foundWords;
    }

    private static void dfs(char[][] board, boolean[][] visited, int row, int col, StringBuilder currentWord,
            Trie trie, Set<String> foundWords) {
        int rows = board.length;
        int columns = board[0].length;

        if (row < 0 || row >= rows || col < 0 || col >= columns || visited[row][col]) {
            return;
        }

        currentWord.append(board[row][col]);
        String word = currentWord.toString();

        if (trie.search(word)) {
            foundWords.add(word);
        }

        visited[row][col] = true;
        dfs(board, visited, row - 1, col, currentWord, trie, foundWords); // Up
        dfs(board, visited, row + 1, col, currentWord, trie, foundWords); // Down
        dfs(board, visited, row, col - 1, currentWord, trie, foundWords); // Left
        dfs(board, visited, row, col + 1, currentWord, trie, foundWords); // Right
        dfs(board, visited, row - 1, col - 1, currentWord, trie, foundWords); // Up-Left
        dfs(board, visited, row - 1, col + 1, currentWord, trie, foundWords); // Up-Right
        dfs(board, visited, row + 1, col - 1, currentWord, trie, foundWords); // Down-Left
        dfs(board, visited, row + 1, col + 1, currentWord, trie, foundWords); // Down-Right

        // Backtrack by removing the current letter from the current word and marking
        // the current cell as unvisited
        currentWord.deleteCharAt(currentWord.length() - 1);
        visited[row][col] = false;
    }

    // S73.
    public static String rearrangeString(String input) {
        Map<Character, Integer> charFrequency = new HashMap<>();
        for (char c : input.toCharArray()) {
            charFrequency.put(c, charFrequency.getOrDefault(c, 0) + 1);
        }

        PriorityQueue<Character> priorityQueue = new PriorityQueue<>(
                (a, b) -> charFrequency.get(b) - charFrequency.get(a));
        priorityQueue.addAll(charFrequency.keySet());

        StringBuilder rearrangedString = new StringBuilder();
        while (!priorityQueue.isEmpty()) {
            char currentChar = priorityQueue.remove();

            if (rearrangedString.length() >= 1
                    && rearrangedString.charAt(rearrangedString.length() - 1) == currentChar) {
                if (priorityQueue.isEmpty()) {
                    return "None";
                }
                char nextChar = priorityQueue.remove();
                rearrangedString.append(nextChar);

                int frequency = charFrequency.get(nextChar);
                frequency--;
                if (frequency > 0) {
                    charFrequency.put(nextChar, frequency);
                    priorityQueue.add(nextChar);
                }

                priorityQueue.add(currentChar);
            } else {
                rearrangedString.append(currentChar);

                int frequency = charFrequency.get(currentChar);
                frequency--;
                if (frequency > 0) {
                    charFrequency.put(currentChar, frequency);
                    priorityQueue.add(currentChar);
                }
            }
        }

        return rearrangedString.toString();
    }

    // S74.
    static class PrefixMapSum {
        private Map<String, Integer> map;

        public PrefixMapSum() {
            map = new HashMap<>();
        }

        public void insert(String key, int value) {
            map.put(key, value);
        }

        public int sum(String prefix) {
            int sum = 0;
            for (String key : map.keySet()) {
                if (key.startsWith(prefix)) {
                    sum += map.get(key);
                }
            }
            return sum;
        }
    }

    // S75.
    public static int fib(int n) {
        if (n <= 1) {
            return n;
        }

        int prevPrev = 0;
        int prev = 1;
        int current = 0;

        for (int i = 2; i <= n; i++) {
            current = prevPrev + prev;
            prevPrev = prev;
            prev = current;
        }

        return current;
    }

    // S76.
    static class NonBinaryTreeNode {
        int val;
        List<NonBinaryTreeNode> children;

        NonBinaryTreeNode(int val) {
            this.val = val;
            children = new ArrayList<>();
        }
    }

    public static boolean isSymmetric(NonBinaryTreeNode root) {
        if (root == null) {
            return true;
        }

        return isMirror(root, root);
    }

    private static boolean isMirror(NonBinaryTreeNode node1, NonBinaryTreeNode node2) {
        if (node1 == null && node2 == null) {
            return true;
        }

        if (node1 == null || node2 == null) {
            return false;
        }

        if (node1.val != node2.val) {
            return false;
        }

        if (node1.children.size() != node2.children.size()) {
            return false;
        }

        int size = node1.children.size();
        for (int i = 0; i < size; i++) {
            if (!isMirror(node1.children.get(i), node2.children.get(size - i - 1))) {
                return false;
            }
        }

        return true;
    }

    // S77.
    public static int calculateHIndex(int[] citations) {
        Arrays.sort(citations);

        int hIndex = 0;
        int n = citations.length;

        for (int i = 0; i < n; i++) {
            int smallerCount = Math.min(citations[i], n - i); // Number of papers with at least citations[i] citations
            hIndex = Math.max(hIndex, smallerCount);
        }

        return hIndex;
    }

    // S78.
    public static List<Integer> generatePrimes(int n) {
        boolean[] isComposite = new boolean[n + 1];
        List<Integer> primes = new ArrayList<>();

        for (int i = 2; i <= n; i++) {
            if (!isComposite[i]) {
                primes.add(i);
                for (int j = i * i; j <= n; j += i) {
                    isComposite[j] = true;
                }
            }
        }

        return primes;
    }

    static class PrimeGenerator implements Iterable<Integer> {
        private static List<Integer> primes;

        public PrimeGenerator() {
            primes = new ArrayList<>();
            primes.add(2);
        }

        @Override
        public Iterator<Integer> iterator() {
            return new PrimeIterator();
        }

        private static class PrimeIterator implements Iterator<Integer> {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public Integer next() {
                if (currentIndex < primes.size()) {
                    return primes.get(currentIndex++);
                }

                int currentNumber = primes.get(primes.size() - 1) + 1;
                while (true) {
                    boolean isPrime = true;
                    int sqrt = (int) Math.sqrt(currentNumber);

                    for (int prime : primes) {
                        if (prime > sqrt) {
                            break;
                        }
                        if (currentNumber % prime == 0) {
                            isPrime = false;
                            break;
                        }
                    }

                    if (isPrime) {
                        primes.add(currentNumber);
                        currentIndex++;
                        return currentNumber;
                    }

                    currentNumber++;
                }
            }
        }
    }

    // S79.
    static class BalancedBinaryTree {
        public boolean isBalanced(TreeNode<Integer> root) {
            if (root == null) {
                return true;
            }

            int leftHeight = getHeight(root.left);
            int rightHeight = getHeight(root.right);

            if (Math.abs(leftHeight - rightHeight) > 1) {
                return false;
            }

            return isBalanced(root.left) && isBalanced(root.right);
        }

        private int getHeight(TreeNode<Integer> node) {
            if (node == null) {
                return 0;
            }

            int leftHeight = getHeight(node.left);
            int rightHeight = getHeight(node.right);

            return Math.max(leftHeight, rightHeight) + 1;
        }
    }

    // S80.
    public static List<String> convertToEgyptianFraction(int numerator, int denominator) {
        List<String> egyptianFractions = new ArrayList<>();

        while (numerator != 0) {
            // Calculate the largest unit fraction that is less than or equal to the target
            // fraction
            int unitDenominator = (denominator + numerator - 1) / numerator;

            egyptianFractions.add("1/" + unitDenominator);

            numerator = (numerator * unitDenominator) - denominator;
            denominator *= unitDenominator;
        }

        return egyptianFractions;
    }

    // S81.
    public static int[][] findTransitiveClosure(int[][] graph) {
        int n = graph.length;
        int[][] closure = new int[n][n];

        for (int i = 0; i < n; i++) {
            closure[i][i] = 1; // Each vertex is reachable from itself
            for (int j : graph[i]) {
                closure[i][j] = 1; // Mark the adjacent vertices as reachable
            }
        }

        // Apply the Warshall's algorithm to compute transitive closure
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    closure[i][j] = closure[i][j] | (closure[i][k] & closure[k][j]);
                }
            }
        }

        return closure;
    }

    // S82.
    public static int[] findBounds(int[] arr) {
        int left = 0;
        int right = arr.length - 1;

        while (left < arr.length - 1 && arr[left] <= arr[left + 1]) {
            left++;
        }

        if (left == arr.length - 1) {
            return new int[] { -1, -1 };
        }

        while (right > 0 && arr[right] >= arr[right - 1]) {
            right--;
        }

        // Find the minimum and maximum elements within the subarray
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int i = left; i <= right; i++) {
            min = Math.min(min, arr[i]);
            max = Math.max(max, arr[i]);
        }

        // Expand the bounds to include any elements that need to be sorted
        while (left > 0 && arr[left - 1] > min) {
            left--;
        }

        while (right < arr.length - 1 && arr[right + 1] < max) {
            right++;
        }

        return new int[] { left, right };
    }

    // S83.
    public static void printBoustrophedon(TreeNode<Integer> root) {
        if (root == null) {
            return;
        }

        Queue<TreeNode<Integer>> queue = new LinkedList<>();
        queue.add(root);
        boolean leftToRight = true;

        while (!queue.isEmpty()) {
            List<Integer> levelNodes = new ArrayList<>();
            int levelSize = queue.size();

            for (int i = 0; i < levelSize; i++) {
                TreeNode<Integer> node = queue.poll();

                levelNodes.add(node.val);

                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }

            if (!leftToRight) {
                Collections.reverse(levelNodes);
            }

            for (int val : levelNodes) {
                System.out.print(val + " ");
            }

            leftToRight = !leftToRight;
        }
    }

    // S84.
    public static String buildHuffmanTree(TreeNode<Character> root) {
        StringBuilder result = new StringBuilder();
        traverseHuffmanTree(root, "", result);
        return result.toString();
    }

    private static void traverseHuffmanTree(TreeNode<Character> node, String currentCode, StringBuilder result) {
        if (node == null) {
            return;
        }

        if (node.left == null && node.right == null) {
            result.append(currentCode);
            return;
        }

        traverseHuffmanTree(node.left, currentCode + "0", result);
        traverseHuffmanTree(node.right, currentCode + "1", result);
    }

    // S85.
    public static int[] calculateBonuses(int[] linesOfCode) {
        int n = linesOfCode.length;
        int[] bonuses = new int[n];

        for (int i = 0; i < n; i++) {
            bonuses[i] = 1;
        }

        for (int i = 1; i < n; i++) {
            if (linesOfCode[i] > linesOfCode[i - 1]) {
                // If current employee has more lines of code than the previous one, increase
                // their bonus
                bonuses[i] = bonuses[i - 1] + 1;
            }
        }

        // Check the lines of code in reverse order to handle cases where an employee
        // has fewer lines of code than their next neighbor
        for (int i = n - 2; i >= 0; i--) {
            if (linesOfCode[i] > linesOfCode[i + 1] && bonuses[i] <= bonuses[i + 1]) {
                // If current employee has more lines of code than the next one, and their bonus
                // is less or equal, increase their bonus
                bonuses[i] = bonuses[i + 1] + 1;
            }
        }

        return bonuses;
    }

    // S86.
    public static boolean areAnagrams(String word1, String word2) {
        char[] chars1 = word1.toCharArray();
        char[] chars2 = word2.toCharArray();
        Arrays.sort(chars1);
        Arrays.sort(chars2);
        return Arrays.equals(chars1, chars2);
    }

    public static List<String> findStepWords(String inputWord, List<String> dictionary) {
        List<String> stepWords = new ArrayList<>();
        for (String word : dictionary) {
            for (char c = 'A'; c <= 'Z'; c++) {
                String newWord = inputWord + c;
                if (areAnagrams(newWord, word)) {
                    stepWords.add(word);
                    break;
                }
            }
        }
        return stepWords;
    }

    // S87.
    public static String orientDominos(String dominoes) {
        char[] orientations = dominoes.toCharArray();
        int n = orientations.length;
        int[] forces = new int[n];

        int force = 0; // Cumulative force
        for (int i = 0; i < n; i++) {
            if (orientations[i] == 'R') {
                force = n; // Max force if pushed from the right
            } else if (orientations[i] == 'L') {
                force = 0; // Reset force if pushed from the left
            } else {
                force = Math.max(force - 1, 0); // Decrease force gradually
            }
            forces[i] += force;
        }

        force = 0; // Reset force
        for (int i = n - 1; i >= 0; i--) {
            if (orientations[i] == 'L') {
                force = n; // Max force if pushed from the left
            } else if (orientations[i] == 'R') {
                force = 0; // Reset force if pushed from the right
            } else {
                force = Math.max(force - 1, 0); // Decrease force gradually
            }
            forces[i] -= force;
        }

        StringBuilder result = new StringBuilder();
        for (int forceValue : forces) {
            if (forceValue > 0) {
                result.append('R');
            } else if (forceValue < 0) {
                result.append('L');
            } else {
                result.append('.');
            }
        }

        return result.toString();
    }

    // S88.
    public static int findFixedPoint(int[] nums) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == mid) {
                return mid;
            } else if (nums[mid] < mid) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }

    // S89.
    public static boolean validUtf8(int[] data) {
        int remainingBytes = 0;

        for (int num : data) {
            if (remainingBytes == 0) {
                if ((num >> 5) == 0b110) {
                    remainingBytes = 1;
                } else if ((num >> 4) == 0b1110) {
                    remainingBytes = 2;
                } else if ((num >> 3) == 0b11110) {
                    remainingBytes = 3;
                } else if ((num >> 7) != 0) {
                    return false;
                }
            } else {
                if ((num >> 6) != 0b10) {
                    return false;
                }
                remainingBytes--;
            }
        }

        return remainingBytes == 0;
    }

    // S90.
    public static List<TreeNode<Integer>> generateBSTs(int n) {
        return generateTrees(1, n);
    }

    private static List<TreeNode<Integer>> generateTrees(int start, int end) {
        List<TreeNode<Integer>> trees = new ArrayList<>();

        // Base case: if start > end, return null to represent an empty subtree
        if (start > end) {
            trees.add(null);
            return trees;
        }

        // Generate all possible combinations of left and right subtrees
        for (int i = start; i <= end; i++) {
            List<TreeNode<Integer>> leftSubtrees = generateTrees(start, i - 1);
            List<TreeNode<Integer>> rightSubtrees = generateTrees(i + 1, end);

            // Create all possible combinations of left and right subtrees
            for (TreeNode<Integer> left : leftSubtrees) {
                for (TreeNode<Integer> right : rightSubtrees) {
                    TreeNode<Integer> root = new TreeNode<>(i);
                    root.left = left;
                    root.right = right;
                    trees.add(root);
                }
            }
        }

        return trees;
    }

    // S91.
    public static int countFriendGroups(int[][] adjacencyList) {
        int n = adjacencyList.length;
        boolean[] visited = new boolean[n];
        int count = 0;

        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs(adjacencyList, i, visited);
                count++;
            }
        }

        return count;
    }

    private static void dfs(int[][] adjacencyList, int node, boolean[] visited) {
        visited[node] = true;

        for (int neighbor : adjacencyList[node]) {
            if (!visited[neighbor]) {
                dfs(adjacencyList, neighbor, visited);
            }
        }
    }

    // S92.
    static class Graph {
        private int vertices;
        private List<List<Integer>> adjList;

        public Graph(int vertices) {
            this.vertices = vertices;
            adjList = new ArrayList<>(vertices);
            for (int i = 0; i < vertices; i++) {
                adjList.add(new ArrayList<>());
            }
        }

        public void addEdge(int src, int dest) {
            adjList.get(src).add(dest);
            adjList.get(dest).add(src);
        }

        public boolean containsCycle() {
            boolean[] visited = new boolean[vertices];

            for (int i = 0; i < vertices; i++) {
                if (!visited[i] && hasCycle(i, visited, -1)) {
                    return true;
                }
            }

            return false;
        }

        private boolean hasCycle(int vertex, boolean[] visited, int parent) {
            visited[vertex] = true;

            for (int neighbor : adjList.get(vertex)) {
                if (!visited[neighbor]) {
                    if (hasCycle(neighbor, visited, vertex)) {
                        return true;
                    }
                } else if (neighbor != parent) {
                    // If the neighbor is already visited and not the parent of the current vertex,
                    // a cycle is found.
                    return true;
                }
            }

            return false;
        }
    }

    // S93.
    public static boolean containsPythagoreanTriplet(int[] nums) {
        int n = nums.length;

        for (int i = 0; i < n; i++) {
            nums[i] = nums[i] * nums[i];
        }

        Arrays.sort(nums);

        // Fix one element as a
        for (int i = 0; i < n - 2; i++) {
            // Fix the second element as b and b
            for (int j = i + 1; j < n - 1; j++) {
                int c = nums[i] + nums[j];

                if (binarySearch(nums, c, j + 1, n - 1)) {
                    return true;
                }
            }
        }

        return false;
    }

    private static boolean binarySearch(int[] nums, int target, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return true;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return false;
    }

    // S94.
    public static List<Long> getRegularNumbers(int n) {
        List<Long> regularNumbers = new ArrayList<>();
        regularNumbers.add(1L);
        int i2 = 0, i3 = 0, i5 = 0;

        while (regularNumbers.size() < n) {
            long nextRegularNumber = Math.min(regularNumbers.get(i2) * 2,
                    Math.min(regularNumbers.get(i3) * 3, regularNumbers.get(i5) * 5));
            regularNumbers.add(nextRegularNumber);

            if (nextRegularNumber == regularNumbers.get(i2) * 2) {
                i2++;
            }
            if (nextRegularNumber == regularNumbers.get(i3) * 3) {
                i3++;
            }
            if (nextRegularNumber == regularNumbers.get(i5) * 5) {
                i5++;
            }
        }

        return regularNumbers;
    }

    // S95.
    public static int getRemainingQuxes(char[] quxes) {
        Stack<Character> stack = new Stack<>();

        for (char qux : quxes) {
            if (!stack.isEmpty() && stack.peek() != qux) {
                stack.pop(); // Quxes transform into the third color
            } else {
                stack.push(qux);
            }
        }

        return stack.size();
    }

    // S96.
    public static int longestTwoAppleTrees(int[] trees) {
        int n = trees.length;
        if (n < 3) {
            return n; // Not enough trees to form a portion
        }

        int maxLen = 0;
        Map<Integer, Integer> treeCounts = new HashMap<>();
        int left = 0;
        int right = 0;

        while (right < n) {
            treeCounts.put(trees[right], treeCounts.getOrDefault(trees[right], 0) + 1);

            while (treeCounts.size() > 2) {
                treeCounts.put(trees[left], treeCounts.get(trees[left]) - 1);
                if (treeCounts.get(trees[left]) == 0) {
                    treeCounts.remove(trees[left]);
                }
                left++;
            }

            maxLen = Math.max(maxLen, right - left + 1);
            right++;
        }

        return maxLen;
    }

    // S97.
    private static class Candidate implements Comparable<Candidate> {
        private final int id;
        private final int count;

        public Candidate(int id, int count) {
            this.id = id;
            this.count = count;
        }

        public int getId() {
            return id;
        }

        public int getCount() {
            return count;
        }

        @Override
        public int compareTo(Candidate other) {
            return Integer.compare(this.count, other.count);
        }
    }

    // S98.
    public static double calculateAngle(String time) {
        String[] parts = time.split(":");
        int hour = Integer.parseInt(parts[0]);
        int minute = Integer.parseInt(parts[1]);

        double hourAngle = (hour % 12 + minute / 60.0) * 30;
        double minuteAngle = minute * 6;

        double angle = Math.abs(hourAngle - minuteAngle);

        return Math.min(angle, 360 - angle);
    }

    // S99.
    public static ListNode removeZeroSumSublists(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        // Create a prefix sum map to track cumulative sums and their corresponding
        // nodes
        Map<Integer, ListNode> prefixSumMap = new HashMap<>();
        int prefixSum = 0;
        ListNode curr = dummy;

        while (curr != null) {
            prefixSum += curr.val;

            if (prefixSumMap.containsKey(prefixSum)) {
                // Remove the nodes between the previous occurrence of the prefix sum and the
                // current node
                ListNode prev = prefixSumMap.get(prefixSum).next;
                int sum = prefixSum + prev.val;

                while (prev != curr) {
                    prefixSumMap.remove(sum);
                    prev = prev.next;
                    sum += prev.val;
                }

                prefixSumMap.get(prefixSum).next = curr.next;
            } else {
                prefixSumMap.put(prefixSum, curr);
            }

            curr = curr.next;
        }

        return dummy.next;
    }

    // S100.
    private static TreeNode<Integer> insert(TreeNode<Integer> root, int val) {
        if (root == null) {
            return new TreeNode<>(val);
        }

        if (val < root.val) {
            root.left = insert(root.left, val);
        } else {
            root.right = insert(root.right, val);
        }

        return root;
    }

    public static Integer findFloor(TreeNode<Integer> root, int target) {
        if (root == null) {
            return null;
        }

        if (root.val == target) {
            return root.val;
        }

        if (root.val > target) {
            return findFloor(root.left, target);
        }

        Integer floor = findFloor(root.right, target);
        return (floor != null && floor <= target) ? floor : root.val;
    }

    public static Integer findCeiling(TreeNode<Integer> root, int target) {
        if (root == null) {
            return null;
        }

        if (root.val == target) {
            return root.val;
        }

        if (root.val < target) {
            return findCeiling(root.right, target);
        }

        Integer ceiling = findCeiling(root.left, target);
        return (ceiling != null && ceiling >= target) ? ceiling : root.val;
    }

    // S101.
    public static int countSetBits(int N) {
        int count = 0;

        for (int i = 1; i <= N; i++) {
            count += countSetBitsUtil(i);
        }

        return count;
    }

    public static int countSetBitsUtil(int num) {
        int count = 0;

        while (num > 0) {
            if ((num & 1) == 1) {
                count++;
            }

            num >>= 1;
        }

        return count;
    }

    // S102.
    public static int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left - 1;
    }

    // S103.
    public static int countWaysToCoverBoard(int N) {
        if (N == 0) {
            return 1;
        }

        int[] dp = new int[N + 1];
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= N; i++) {
            // dp[i-1]: the number of ways to cover the board by extending the previous
            // width by a 2x1 domino
            // dp[i-2]: the number of ways to cover the board by extending the previous
            // width by a 2x2 tromino
            // 2*dp[i-2]: the number of ways to cover the board by placing two 1x2 trominoes
            // vertically
            dp[i] = dp[i - 1] + dp[i - 2] + 2 * dp[i - 2];
        }

        return dp[N]; // 1 1 4 7 19 by increasing N by 1
    }

    // S104.
    public static boolean isToeplitzMatrix(int[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;

        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < columns; j++) {
                if (matrix[i][j] != matrix[i - 1][j - 1]) {
                    return false;
                }
            }
        }

        return true;
    }

    // S105.
    public static int minimizeSteps(int[] mice, int[] holes) {
        Arrays.sort(mice);
        Arrays.sort(holes);

        int maxSteps = 0;

        for (int i = 0; i < mice.length; i++) {
            int steps = Math.abs(mice[i] - holes[i]);
            maxSteps = Math.max(maxSteps, steps);
        }

        return maxSteps;
    }

    // S106.
    static class UnitConverter {
        private Map<String, Map<String, Double>> conversionGraph;

        public UnitConverter() {
            this.conversionGraph = new HashMap<>();
        }

        public void addUnitConversion(String fromUnit, String toUnit, double conversionRate) {
            conversionGraph.putIfAbsent(fromUnit, new HashMap<>());
            conversionGraph.putIfAbsent(toUnit, new HashMap<>());

            conversionGraph.get(fromUnit).put(toUnit, conversionRate);
            conversionGraph.get(toUnit).put(fromUnit, 1.0 / conversionRate);
        }

        public double convert(double quantity, String fromUnit, String toUnit) {
            if (fromUnit.equals(toUnit)) {
                return quantity;
            }

            if (!conversionGraph.containsKey(fromUnit) || !conversionGraph.containsKey(toUnit)) {
                throw new IllegalArgumentException("Conversion not defined between " + fromUnit + " and " + toUnit);
            }

            Map<String, Double> distances = new HashMap<>();
            distances.put(fromUnit, 1.0);

            Queue<String> queue = new LinkedList<>();
            queue.offer(fromUnit);

            while (!queue.isEmpty()) {
                String currentUnit = queue.poll();
                double currentDistance = distances.get(currentUnit);

                if (currentUnit.equals(toUnit)) {
                    return quantity * currentDistance;
                }

                Map<String, Double> neighbors = conversionGraph.get(currentUnit);

                for (String neighbor : neighbors.keySet()) {
                    if (!distances.containsKey(neighbor)) {
                        double neighborDistance = currentDistance * neighbors.get(neighbor);
                        distances.put(neighbor, neighborDistance);
                        queue.offer(neighbor);
                    }
                }
            }

            throw new IllegalArgumentException("Conversion not possible from " + fromUnit + " to " + toUnit);
        }
    }

    // S107.
    static public TreeNode<Integer> mergeTrees(TreeNode<Integer> t1, TreeNode<Integer> t2) {
        if (t1 == null && t2 == null) {
            return null;
        }

        int val1 = (t1 != null) ? t1.val : 0;
        int val2 = (t2 != null) ? t2.val : 0;

        TreeNode<Integer> newNode = new TreeNode<>(val1 + val2);
        newNode.left = mergeTrees((t1 != null) ? t1.left : null, (t2 != null) ? t2.left : null);
        newNode.right = mergeTrees((t1 != null) ? t1.right : null, (t2 != null) ? t2.right : null);

        return newNode;
    }

    // S108.
    public static int countPairs(int M, int N) {
        int count = 0;

        for (int a = 1; a <= M / 2; a++) {
            int b = M - a;
            if ((a ^ b) == N) {
                count++;
            }
        }

        return count;
    }

    // S109.
    public static boolean canReach24(int[] nums) {
        return canReachTarget(nums, 24);
    }

    private static boolean canReachTarget(int[] nums, int target) {
        if (nums.length == 1) {
            return nums[0] == target;
        }

        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (i != j) {
                    int[] remaining = getRemainingArray(nums, i, j);

                    if (canReachTarget(applyOperators(nums[i], nums[j], remaining), target)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private static int[] applyOperators(int a, int b, int[] remaining) {
        int[] result = new int[remaining.length + 1];

        for (int i = 0; i < remaining.length; i++) {
            result[i] = remaining[i];
        }

        result[result.length - 1] = a + b;

        if (canReachTarget(result, 24)) {
            return result;
        }

        result[result.length - 1] = a - b;

        if (canReachTarget(result, 24)) {
            return result;
        }

        result[result.length - 1] = a * b;

        if (canReachTarget(result, 24)) {
            return result;
        }

        if (b != 0 && a % b == 0) {
            result[result.length - 1] = a / b;

            if (canReachTarget(result, 24)) {
                return result;
            }
        }

        return remaining;
    }

    private static int[] getRemainingArray(int[] nums, int i, int j) {
        int[] remaining = new int[nums.length - 2];
        int index = 0;

        for (int k = 0; k < nums.length; k++) {
            if (k != i && k != j) {
                remaining[index++] = nums[k];
            }
        }

        return remaining;
    }

    // S110.
    public static boolean hasThreeSum(int[] nums, int k) {
        HashSet<Integer> set = new HashSet<>();

        for (int i = 0; i < nums.length - 2; i++) {
            int target = k - nums[i];
            set.clear();

            for (int j = i + 1; j < nums.length; j++) {
                if (set.contains(target - nums[j])) {
                    return true;
                }
                set.add(nums[j]);
            }
        }

        return false;
    }

    // S111.
    static class Point {
        int x;
        int y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            return "(" + x + ", " + y + ")";
        }
    }

    public static Point[] findClosestPoints(Point[] points) {
        if (points == null || points.length < 2) {
            return null;
        }

        Point[] result = new Point[2];
        double minDistance = Double.POSITIVE_INFINITY;

        for (int i = 0; i < points.length; i++) {
            for (int j = i + 1; j < points.length; j++) {
                double distance = calculateDistance(points[i], points[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    result[0] = points[i];
                    result[1] = points[j];
                }
            }
        }

        return result;
    }

    private static double calculateDistance(Point p1, Point p2) {
        int dx = p1.x - p2.x;
        int dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    // S112.
    public static int findMaxPackedWords(char[][] board, String[] dictionary) {
        Arrays.sort(dictionary, Comparator.comparingInt(String::length));

        int count = 0;

        boolean[][] visited = new boolean[board.length][board[0].length];
        for (String word : dictionary) {
            if (isWordPacked(board, visited, word)) {
                count++;
            }
        }

        return count;
    }

    private static boolean isWordPacked(char[][] board, boolean[][] visited, String word) {
        int rows = board.length;
        int cols = board[0].length;
        char firstChar = word.charAt(0);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == firstChar && !visited[i][j]) {
                    visited[i][j] = true;

                    if (dfs(board, visited, word.substring(1), i, j)) {
                        return true;
                    }

                    visited[i][j] = false;
                }
            }
        }

        return false;
    }

    private static boolean dfs(char[][] board, boolean[][] visited, String word, int row, int col) {
        int rows = board.length;
        int cols = board[0].length;

        if (word.length() == 0) {
            return true; // All characters of the word have been visited
        }

        int[] dx = { -1, 1, 0, 0 };
        int[] dy = { 0, 0, -1, 1 };

        for (int i = 0; i < 4; i++) {
            int newRow = row + dx[i];
            int newCol = col + dy[i];

            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols && !visited[newRow][newCol]
                    && board[newRow][newCol] == word.charAt(0)) {
                visited[newRow][newCol] = true;

                if (dfs(board, visited, word.substring(1), newRow, newCol)) {
                    return true;
                }

                visited[newRow][newCol] = false;
            }
        }

        return false;
    }

    // S113.
    public static String getLexicographicallySmallestString(String input, int k) {
        String smallestString = input;

        for (int i = 0; i < k; i++) {
            String currentString = input.substring(i + 1) + input.substring(0, i + 1);
            if (currentString.compareTo(smallestString) < 0) {
                smallestString = currentString;
            }
        }

        return smallestString;
    }

    // S114.
    static class TernarySearchTree {
        private Node root;

        private static class Node {
            char character;
            Node left, middle, right;
            boolean isEndOfWord;

            Node(char character) {
                this.character = character;
            }
        }

        public void insert(String word) {
            root = insert(root, word, 0);
        }

        private Node insert(Node node, String word, int index) {
            char currentChar = word.charAt(index);
            if (node == null) {
                node = new Node(currentChar);
            }

            if (currentChar < node.character) {
                node.left = insert(node.left, word, index);
            } else if (currentChar > node.character) {
                node.right = insert(node.right, word, index);
            } else {
                if (index < word.length() - 1) {
                    node.middle = insert(node.middle, word, index + 1);
                } else {
                    node.isEndOfWord = true;
                }
            }

            return node;
        }

        public boolean search(String word) {
            return search(root, word, 0);
        }

        private boolean search(Node node, String word, int index) {
            if (node == null) {
                return false;
            }

            char currentChar = word.charAt(index);
            if (currentChar < node.character) {
                return search(node.left, word, index);
            } else if (currentChar > node.character) {
                return search(node.right, word, index);
            } else {
                if (index == word.length() - 1) {
                    return node.isEndOfWord;
                } else {
                    return search(node.middle, word, index + 1);
                }
            }
        }
    }

    // S115.
    public static boolean isCrosswordGrid(char[][] grid) {
        int n = grid.length;

        // Check rule 1: Every white square must be part of an "across" word and a
        // "down" word
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 'W') {
                    if (!hasAcrossWord(grid, i, j) && !hasDownWord(grid, i, j)) {
                        return false;
                    }
                }
            }
        }

        // Check rule 2: No word can be fewer than three letters long
        if (!checkWordLengths(grid)) {
            return false;
        }

        // Check rule 3: Every white square must be reachable from every other white
        // square
        boolean[][] visited = new boolean[n][n];
        int startX = -1;
        int startY = -1;
        outer: for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 'W') {
                    startX = i;
                    startY = j;
                    break outer;
                }
            }
        }

        dfs(grid, visited, startX, startY);

        // Check if all white squares were visited
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 'W' && !visited[i][j]) {
                    return false;
                }
            }
        }

        // Check rule 4: The grid is rotationally symmetric
        return isSymmetric(grid);
    }

    private static boolean hasAcrossWord(char[][] grid, int row, int col) {
        int n = grid.length;
        return (col > 0 && grid[row][col - 1] == 'W') || (col < n - 1 && grid[row][col + 1] == 'W');
    }

    private static boolean hasDownWord(char[][] grid, int row, int col) {
        int n = grid.length;
        return (row > 0 && grid[row - 1][col] == 'W') || (row < n - 1 && grid[row + 1][col] == 'W');
    }

    private static boolean checkWordLengths(char[][] grid) {
        int n = grid.length;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 'W') {
                    int length = 1;

                    if (j > 0 && grid[i][j - 1] == 'W') {
                        length++;
                        if (j > 1 && grid[i][j - 2] == 'W') {
                            length++;
                        }
                    }

                    if (j < n - 1 && grid[i][j + 1] == 'W') {
                        length++;
                        if (j < n - 2 && grid[i][j + 2] == 'W') {
                            length++;
                        }
                    }

                    if (i > 0 && grid[i - 1][j] == 'W') {
                        length++;
                        if (i > 1 && grid[i - 2][j] == 'W') {
                            length++;
                        }
                    }

                    if (i < n - 1 && grid[i + 1][j] == 'W') {
                        length++;
                        if (i < n - 2 && grid[i + 2][j] == 'W') {
                            length++;
                        }
                    }

                    if (length < 3) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private static void dfs(char[][] grid, boolean[][] visited, int row, int col) {
        int n = grid.length;
        if (row < 0 || row >= n || col < 0 || col >= n || visited[row][col] || grid[row][col] == 'B') {
            return;
        }
        visited[row][col] = true;
        dfs(grid, visited, row - 1, col);
        dfs(grid, visited, row + 1, col);
        dfs(grid, visited, row, col - 1);
        dfs(grid, visited, row, col + 1);
    }

    private static boolean isSymmetric(char[][] grid) {
        int n = grid.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != grid[n - 1 - i][n - 1 - j]) {
                    return false;
                }
            }
        }
        return true;
    }

    // S116.
    public static String getOriginalDigits(String s) {
        Map<Character, Integer> countMap = new HashMap<>();

        for (char c : s.toCharArray()) {
            countMap.put(c, countMap.getOrDefault(c, 0) + 1);
        }

        String[] digitToChar = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
        int[] digitOrder = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 };
        char[] uniqueChars = { 'z', 'o', 'w', 't', 'u', 'f', 'x', 's', 'g', 'i' };

        StringBuilder result = new StringBuilder();
        for (int digit : digitOrder) {
            char uniqueChar = uniqueChars[digit];
            int count = countMap.getOrDefault(uniqueChar, 0);

            for (int i = 0; i < count; i++) {
                result.append(digit);
            }

            for (char c : digitToChar[digit].toCharArray()) {
                countMap.put(c, countMap.getOrDefault(c, 0) - count);
            }
        }

        char[] resultArray = result.toString().toCharArray();
        Arrays.sort(resultArray);
        return new String(resultArray);
    }

    // S117.
    public static List<String> findStrobogrammaticNumbers(int n) {
        return findStrobogrammaticNumbersHelper(n, n);
    }

    private static List<String> findStrobogrammaticNumbersHelper(int n, int m) {
        if (n == 0) {
            return new ArrayList<>(Arrays.asList(""));
        }
        if (n == 1) {
            return new ArrayList<>(Arrays.asList("0", "1", "8"));
        }

        List<String> result = new ArrayList<>();
        List<String> inner = findStrobogrammaticNumbersHelper(n - 2, m);

        for (String num : inner) {
            if (n != m) {
                result.add("0" + num + "0");
            }
            result.add("1" + num + "1");
            result.add("6" + num + "9");
            result.add("8" + num + "8");
            result.add("9" + num + "6");
        }

        return result;
    }

    // S118.
    public static int calculateTotalActiveTime(String[] data) {
        Map<Integer, Integer> activeTimeMap = new HashMap<>();
        int totalActiveTime = 0;

        for (String entry : data) {
            String[] parts = entry.split(", ");

            int deliveryId = Integer.parseInt(parts[0].substring(1));
            int timestamp = Integer.parseInt(parts[1]);
            String action = parts[2].substring(1, parts[2].length() - 2);

            if (action.equals("pickup")) {
                activeTimeMap.put(deliveryId, timestamp);
            } else if (action.equals("dropoff")) {
                if (activeTimeMap.containsKey(deliveryId)) {
                    int pickupTimestamp = activeTimeMap.get(deliveryId);
                    totalActiveTime += timestamp - pickupTimestamp;
                    System.out.println(totalActiveTime);
                    activeTimeMap.remove(deliveryId);
                }
            }
        }

        return totalActiveTime;
    }

    // S119.
    public static int countDigits(int number) {
        String numberString = String.valueOf(number);
        return numberString.length();
    }

    // S120.
    public static int[] findClosestCoin(int[] currentPosition, char[][] map) {
        int minDistance = Integer.MAX_VALUE;
        int[] closestCoin = null;

        for (int i = 0; i < map.length; i++) {
            for (int j = 0; j < map[i].length; j++) {
                if (map[i][j] == 'o') {
                    int[] coinPosition = { i, j };
                    int distance = calculateManhattanDistance(currentPosition, coinPosition);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestCoin = coinPosition;
                    }
                }
            }
        }

        return closestCoin;
    }

    public static int calculateManhattanDistance(int[] point1, int[] point2) {
        return Math.abs(point1[0] - point2[0]) + Math.abs(point1[1] - point2[1]);
    }

    // S121.
    public static List<String> generateSubsequences(String str) {
        List<String> subsequences = new ArrayList<>();
        generateSubsequencesHelper(str, "", 0, subsequences);
        return subsequences;
    }

    private static void generateSubsequencesHelper(String str, String current, int index, List<String> subsequences) {
        if (index == str.length()) {
            subsequences.add(current);
            return;
        }

        // Exclude current character
        generateSubsequencesHelper(str, current, index + 1, subsequences);
        // Include current character
        generateSubsequencesHelper(str, current + str.charAt(index), index + 1, subsequences);
    }

    // S122.
    private static final String BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    private static String convertHexToBase64(String hexString) {
        byte[] hexBytes = hexStringToByteArray(hexString);
        StringBuilder base64Builder = new StringBuilder();

        int i = 0;
        while (i < hexBytes.length) {
            int byte1 = hexBytes[i++] & 0xFF;
            int byte2 = (i < hexBytes.length) ? hexBytes[i++] & 0xFF : 0;
            int byte3 = (i < hexBytes.length) ? hexBytes[i++] & 0xFF : 0;

            int index1 = byte1 >>> 2;
            int index2 = ((byte1 & 0x03) << 4) | (byte2 >>> 4);
            int index3 = ((byte2 & 0x0F) << 2) | (byte3 >>> 6);
            int index4 = byte3 & 0x3F;

            base64Builder.append(BASE64_CHARS.charAt(index1));
            base64Builder.append(BASE64_CHARS.charAt(index2));
            base64Builder.append(BASE64_CHARS.charAt(index3));
            base64Builder.append(BASE64_CHARS.charAt(index4));
        }

        int paddingLength = 3 - (hexBytes.length % 3);
        for (int j = 0; j < paddingLength; j++) {
            base64Builder.setCharAt(base64Builder.length() - 1 - j, '=');
        }

        return base64Builder.toString();
    }

    private static byte[] hexStringToByteArray(String hexString) {
        int length = hexString.length();
        byte[] byteArray = new byte[length / 2];

        for (int i = 0; i < length; i += 2) {
            byteArray[i / 2] = (byte) ((Character.digit(hexString.charAt(i), 16) << 4)
                    + Character.digit(hexString.charAt(i + 1), 16));
        }

        return byteArray;
    }

    // S123.
    private static String convertBase64ToHex(String base64String) {
        StringBuilder hexBuilder = new StringBuilder();
        int paddingCount = countPadding(base64String);
        int numGroups = (base64String.length() - paddingCount) / 4;

        for (int i = 0; i < numGroups; i++) {
            int[] base64Indexes = new int[4];

            for (int j = 0; j < 4; j++) {
                base64Indexes[j] = BASE64_CHARS.indexOf(base64String.charAt(i * 4 + j));
            }

            int value1 = (base64Indexes[0] << 2) | (base64Indexes[1] >> 4);
            int value2 = ((base64Indexes[1] & 0xF) << 4) | (base64Indexes[2] >> 2);
            int value3 = ((base64Indexes[2] & 0x3) << 6) | base64Indexes[3];

            hexBuilder.append(byteToHex(value1)).append(byteToHex(value2)).append(byteToHex(value3));
        }

        if (paddingCount == 1) {
            int[] base64Indexes = new int[4];

            for (int j = 0; j < 3; j++) {
                base64Indexes[j] = BASE64_CHARS.indexOf(base64String.charAt(numGroups * 4 + j));
            }
            base64Indexes[3] = 0; // Padding is treated as 0

            int value1 = (base64Indexes[0] << 2) | (base64Indexes[1] >> 4);
            int value2 = ((base64Indexes[1] & 0xF) << 4) | (base64Indexes[2] >> 2);

            hexBuilder.append(byteToHex(value1)).append(byteToHex(value2));
        } else if (paddingCount == 2) {
            int[] base64Indexes = new int[4];

            for (int j = 0; j < 2; j++) {
                base64Indexes[j] = BASE64_CHARS.indexOf(base64String.charAt(numGroups * 4 + j));
            }
            base64Indexes[2] = base64Indexes[3] = 0; // Padding is treated as 0

            int value1 = (base64Indexes[0] << 2) | (base64Indexes[1] >> 4);

            hexBuilder.append(byteToHex(value1));
        }

        return hexBuilder.toString();
    }

    private static int countPadding(String base64) {
        int paddingCount = 0;

        for (int i = base64.length() - 1; i >= 0; i--) {
            if (base64.charAt(i) == '=') {
                paddingCount++;
            } else {
                break;
            }
        }

        return paddingCount;
    }

    private static String byteToHex(int value) {
        return String.format("%02x", value);
    }

    // S124.
    public static String sortStringByFrequency(String str) {
        Map<Character, Integer> frequencyMap = new HashMap<>();

        for (char ch : str.toCharArray()) {
            frequencyMap.put(ch, frequencyMap.getOrDefault(ch, 0) + 1);
        }

        List<Character> characters = new ArrayList<>(frequencyMap.keySet());

        characters.sort((a, b) -> frequencyMap.get(b) - frequencyMap.get(a));

        StringBuilder sortedString = new StringBuilder();
        for (char ch : characters) {
            int frequency = frequencyMap.get(ch);
            for (int i = 0; i < frequency; i++) {
                sortedString.append(ch);
            }
        }

        return sortedString.toString();
    }

    // S125.
    public static boolean hasPathSum(TreeNode<Integer> root, int k) {
        if (root == null) {
            return false;
        }
        return hasPathSumHelper(root, k);
    }

    private static boolean hasPathSumHelper(TreeNode<Integer> node, int sum) {
        if (node == null) {
            return false;
        }
        if (node.left == null && node.right == null && node.val == sum) {
            return true;
        }

        return hasPathSumHelper(node.left, sum - node.val) || hasPathSumHelper(node.right, sum - node.val);
    }

    // S126.
    public static int rand5() {
        return (int) (Math.random() * 5) + 1;
    }

    public static int rand7() {
        while (true) {
            int num = (rand5() - 1) * 5 + (rand5() - 1);

            if (num < 21) {
                return (num % 7) + 1;
            }
        }
    }

    // S127.
    public static int minStepsToOne(int N) {
        if (N <= 1) {
            return 0;
        }

        int[] dp = new int[N + 1];

        for (int i = 2; i <= N; i++) {
            // Initialize the minimum steps for i to be the steps for i-1 plus 1
            dp[i] = dp[i - 1] + 1;

            // Check if i can be factored into two numbers a and b
            for (int j = 2; j * j <= i; j++) {
                if (i % j == 0) {
                    // Calculate the steps for i using the larger factor of a and b
                    dp[i] = Math.min(dp[i], dp[Math.max(j, i / j)] + 1);
                }
            }
        }

        return dp[N];
    }

    public static void main(String[] args) throws InterruptedException {
        /*
         * Q1.
         * Given a list of numbers and a number k, return whether any two numbers from
         * the list add up to k.
         * For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is
         * 17.
         * Bonus: Can you do this in one pass?
         */
        System.out.println("========= Q1 =========");
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
        System.out.println("========= Q2 =========");
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
        System.out.println("========= Q3 =========");
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
        System.out.println("========= Q4 =========");
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
        System.out.println("========= Q5 =========");
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
        System.out.println("========= Q6 =========");
        // Solution implemented in S6 OrderLog class

        /*
         * Q7.
         * Given a string of round, curly, and square open and closing brackets, return
         * whether the brackets are balanced (well-formed).
         * For example, given the string "([])[]({})", you should return true.
         * Given the string "([)]" or "((()", you should return false.
         */
        System.out.println("========= Q7 =========");
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
        System.out.println("========= Q8 =========");
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
        System.out.println("========= Q9 =========");
        String stringOne = "kitten";
        String stringTwo = "sitting";
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
        int editDistanceOfTwoStrings = editDistance(stringOne, stringTwo);
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
        System.out.println("========= Q10 =========");
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
        System.out.println("========= Q11 =========");
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
        System.out.println("========= Q12 =========");
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
        System.out.println("========= Q13 =========");
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
        System.out.println("========= Q14 =========");
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
        System.out.println("========= Q15 =========");
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
        System.out.println("========= Q16 =========");
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
        System.out.println("========= Q17 =========");
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
        System.out.println("========= Q18 =========");
        int[] listOfIntegers = { -10, -10, 5, 2 };
        int largestProduct = largestProductOfThree(listOfIntegers);
        System.out.println(largestProduct);

        /*
         * Q19.
         * A number is considered perfect if its digits sum up to exactly 10.
         * Given a positive integer n, return the n-th perfect number.
         * For example, given 1, you should return 19. Given 2, you should return 28.
         */
        System.out.println("========= Q19 =========");
        int n = 1;
        int nthPerfectNumber = nthPerfectNumber(n);
        System.out.println(nthPerfectNumber);

        /*
         * Q20.
         * Given the head of a singly linked list, reverse it in-place.
         */
        System.out.println("========= Q20 =========");
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
        System.out.println("========= Q21 =========");
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
        System.out.println("========= Q22 =========");
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
        System.out.println("========= Q23 =========");
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
        System.out.println("========= Q24 =========");

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
        System.out.println("========= Q25 =========");
        // It will print '9' tem times as the lambda functions created using `lambda: i`
        // all reference the same variable 'i'
        // CORRECTION: functions.append(lambda x=i: x)

        /*
         * Q26.
         * Given a binary tree of integers, find the maximum path sum between two nodes.
         * The path must go through at least one node, and does not need to go through
         * the root.
         */
        System.out.println("========= Q26 =========");
        // Solution implemented in S26 (NOT tested)

        /*
         * Q27.
         * Given a number in the form of a list of digits, return all possible
         * permutations.
         * For example, given [1,2,3], return
         * [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]].
         */
        System.out.println("========= Q27 =========");
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
        System.out.println("========= Q28 =========");
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
        System.out.println("========= Q29 =========");
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
        System.out.println("========= Q30 =========");
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an even number greater than 2: ");
        int num = sc.nextInt();
        int[] primes = getPrimes(num);
        System.out.printf("%d + %d = %d\n", primes[0], primes[1], num);
        sc.close();

        /*
         * Q31.
         * Determine whether a doubly linked list is a palindrome. What if it’s singly
         * linked?
         * For example, 1 -> 4 -> 3 -> 4 -> 1 returns True while 1 -> 4 returns False.
         */
        System.out.println("========= Q31 =========");
        DoublyLinkedList list1 = new DoublyLinkedList();
        list1.add(1);
        list1.add(4);
        list1.add(3);
        list1.add(4);
        list1.add(1);
        System.out.println(list1.isPalindrome());

        DoublyLinkedList list2 = new DoublyLinkedList();
        list2.add(1);
        list2.add(4);
        System.out.println(list2.isPalindrome());

        SinglyLinkedList singlyLinkedList1 = new SinglyLinkedList();
        list1.add(1);
        list1.add(4);
        list1.add(3);
        list1.add(4);
        list1.add(1);
        System.out.println(singlyLinkedList1.isPalindrome());

        SinglyLinkedList singlyLinkedList2 = new SinglyLinkedList();
        list2.add(1);
        list2.add(4);
        System.out.println(singlyLinkedList2.isPalindrome());

        /*
         * Q32.
         * Given a function f, and N return a debounced f of N milliseconds.
         * That is, as long as the debounced f continues to be invoked, f itself will
         * not be called for N milliseconds.
         */
        System.out.println("========= Q32 =========");
        Debounce debounce = new Debounce(() -> System.out.println("Hello"), 1000);
        for (int i = 0; i < 10; i++) {
            debounce.execute();
            TimeUnit.MILLISECONDS.sleep(500);
        }

        /*
         * Q33.
         * Print the nodes in a binary tree level-wise. For example, the following
         * should print 1, 2, 3, 4, 5.
         * "   1       "
         * "  / \      "
         * " 2   3     "
         * "    / \    "
         * "   4   5   "
         */
        System.out.println("========= Q33 =========");
        TreeNode<Integer> binaryToPrint = new TreeNode<>(1);
        binaryToPrint.left = new TreeNode<>(2);
        binaryToPrint.right = new TreeNode<>(3);
        binaryToPrint.right.left = new TreeNode<>(4);
        binaryToPrint.right.right = new TreeNode<>(5);

        printLevelOrder(root);

        /*
         * Q34.
         * Given two strings A and B, return whether or not A can be shifted some number
         * of times to get B.
         * For example, if A is abcde and B is cdeab, return true. If A is abc and B is
         * acb, return false.
         */
        System.out.println("========= Q34 =========");
        String A = "abcde";
        String B = "cdeab";
        boolean shiftable1 = canShift(A, B);
        System.out.println(shiftable1);

        String C = "abc";
        String D = "acb";
        boolean shiftable2 = canShift(C, D);
        System.out.println(shiftable2);

        /*
         * Q35.
         * Given a binary tree, return the level of the tree with minimum sum.
         */
        System.out.println("========= Q35 =========");
        TreeNode<Integer> minTreeSum = new TreeNode<>(11);
        minTreeSum.left = new TreeNode<>(2);
        minTreeSum.right = new TreeNode<>(3);

        minTreeSum.left.left = new TreeNode<>(2);
        minTreeSum.left.right = new TreeNode<>(2);
        minTreeSum.right.left = new TreeNode<>(4);
        minTreeSum.right.right = new TreeNode<>(5);

        System.out.print("Minimum Sum Tree Level: ");
        System.out.println(minLevelSum(minTreeSum));

        /*
         * Q36.
         * Given a sorted list of integers, square the elements and give the output in
         * sorted order.
         * For example, given [-9, -2, 0, 2, 3], return [0, 4, 4, 9, 81].
         */
        System.out.println("========= Q36 =========");
        int[] numArr = { -9, -2, 0, 2, 3 };
        int[] squaredArr = sortedSquares(numArr);

        System.out.print("Squared numbers in Array: ");
        for (int i = 0; i < squaredArr.length; i++) {
            System.out.println(squaredArr[i]);
        }

        /*
         * Q37.
         * You have n fair coins and you flip them all at the same time. Any that come
         * up tails you set aside. The ones that come up heads you flip again. How many
         * rounds do you expect to play before only one coin remains?
         * Write a function that, given n, returns the number of rounds you'd expect to
         * play until one coin remains.
         */
        System.out.println("========= Q37 =========");
        int numRounds = numRounds(10);
        System.out.print("Number of rounds until one coin remaining: ");
        System.out.println(numRounds);

        /*
         * Q38.
         * Given the root of a binary search tree, and a target K, return two nodes in
         * the tree whose sum equals K.
         * For example, given the following tree and K of 20
         * "    10         "
         * "   /   \       "
         * " 5      15     "
         * "       /  \    "
         * "     11    15  "
         * Return the nodes 5 and 15.
         */
        System.out.println("========= Q38 =========");
        TreeNode<Integer> findTwoSumBinaryTree = new TreeNode<>(10);
        findTwoSumBinaryTree.left = new TreeNode<>(5);
        findTwoSumBinaryTree.right = new TreeNode<>(15);
        findTwoSumBinaryTree.right.left = new TreeNode<>(11);
        findTwoSumBinaryTree.right.right = new TreeNode<>(15);

        TreeNode<Integer>[] twoNodesForK = findTarget(findTwoSumBinaryTree, 20);
        System.out.println(twoNodesForK[0].val);
        System.out.println(twoNodesForK[1].val);

        /*
         * Q39.
         * Let's represent an integer in a linked list format by having each node
         * represent a digit in the number. The nodes make up the number in reversed
         * order.
         * For example, the following linked list:
         * 1 -> 2 -> 3 -> 4 -> 5
         * is the number 54321.
         * Given two linked lists in this format, return their sum in the same linked
         * list format.
         * For example, given
         * 9 -> 9
         * 5 -> 2
         * return 124 (99 + 25) as:
         * 4 -> 2 -> 1
         */
        System.out.println("========= Q39 =========");
        ListNode numList1 = new ListNode(9);
        numList1.next = new ListNode(9);
        ListNode numList2 = new ListNode(5);
        numList2.next = new ListNode(2);

        ListNode addNumList = addTwoNumbers(numList1, numList2);
        while (addNumList != null) {
            System.out.println(addNumList.val);
            addNumList = addNumList.next;
        }

        /*
         * Q40.
         * Design and implement a HitCounter class that keeps track of requests (or
         * hits). It should support the following operations:
         * record(timestamp): records a hit that happened at timestamp
         * total(): returns the total number of hits recorded
         * range(lower, upper): returns the number of hits that occurred between
         * timestamps lower and upper (inclusive)
         * Follow-up: What if our system has limited memory?
         */
        System.out.println("========= Q40 =========");
        HitCounter hitCounter = new HitCounter();
        hitCounter.record(System.currentTimeMillis());
        Thread.sleep(100);
        long lower = System.currentTimeMillis();
        hitCounter.record(lower);
        Thread.sleep(100);
        hitCounter.record(System.currentTimeMillis());
        Thread.sleep(100);
        long upper = System.currentTimeMillis();
        hitCounter.record(upper);
        Thread.sleep(100);
        hitCounter.record(System.currentTimeMillis());

        System.out.println("Total hits: " + hitCounter.total());
        System.out.println("Hits between lower and upper: " + hitCounter.range(lower, upper));

        /*
         * Q41.
         * You have a large array with most of the elements as zero.
         * Use a more space-efficient data structure, SparseArray, that implements the
         * same interface:
         * init(arr, size): initialize with the original large array and size.
         * set(i, val): updates index at i with val.
         * get(i): gets the value at index i.
         */
        System.out.println("========= Q41 =========");
        int[] sparseArr = { 1, 0, 0, 0, 1, 0, 0, 1, 0, 1 };
        SparseArray sparseArray = new SparseArray(sparseArr, 10);
        sparseArray.set(0, 2);
        sparseArray.set(1, 3);
        System.out.print("First Element in Sparse Array: ");
        System.out.println(sparseArray.get(0));
        System.out.print("Second Element in Sparse Array: ");
        System.out.println(sparseArray.get(1));

        /*
         * Q42.
         * Given a binary tree, find a minimum path sum from root to a leaf.
         * For example, the minimum path in this tree is [10, 5, 1, -1], which has sum
         * 15.
         * "  10       "
         * " /  \      "
         * "5    5     "
         * " \     \   "
         * "   2    1  "
         * "       /   "
         * "     -1    "
         */
        System.out.println("========= Q42 =========");
        TreeNode<Integer> binaryTreeForMinSumPath = new TreeNode<>(10);
        binaryTreeForMinSumPath.left = new TreeNode<>(5);
        binaryTreeForMinSumPath.right = new TreeNode<>(5);
        binaryTreeForMinSumPath.left.right = new TreeNode<>(2);
        binaryTreeForMinSumPath.right.right = new TreeNode<>(1);
        binaryTreeForMinSumPath.right.right.left = new TreeNode<>(-1);

        MinimumPathSum minimumPathSum = new MinimumPathSum();
        int minSum = minimumPathSum.minPathSum(binaryTreeForMinSumPath);
        System.out.println("Minimum path sum: " + minSum);

        /*
         * Q43.
         * Given the head of a singly linked list, swap every two nodes and return its
         * head.
         * For example, given 1 -> 2 -> 3 -> 4, return 2 -> 1 -> 4 -> 3.
         */
        System.out.println("========= Q43 =========");
        ListNode linkedListToSwapEveryTwo = new ListNode(1);
        linkedListToSwapEveryTwo.next = new ListNode(2);
        linkedListToSwapEveryTwo.next.next = new ListNode(3);
        linkedListToSwapEveryTwo.next.next.next = new ListNode(4);

        SwapNodesInPairs swapNodesInPairs = new SwapNodesInPairs();
        ListNode swappedList = swapNodesInPairs.swapPairs(linkedListToSwapEveryTwo);

        ListNode current = swappedList;
        while (current != null) {
            System.out.print(current.val + " ");
            current = current.next;
        }

        /*
         * Q44.
         * Implement a stack API using only a heap. A stack implements the following
         * methods:
         * push(item), which adds an element to the stack
         * pop(), which removes and returns the most recently added element (or throws
         * an error if there is nothing on the stack)
         * Recall that a heap has the following operations:
         * push(item), which adds a new key to the heap
         * pop(), which removes and returns the max value of the heap
         */
        System.out.println("========= Q44 =========");
        StackUsingHeap stackUsingHeap = new StackUsingHeap();

        stackUsingHeap.push(1);
        stackUsingHeap.push(2);
        stackUsingHeap.push(3);

        System.out.println(stackUsingHeap.pop()); // Output: 3
        System.out.println(stackUsingHeap.pop()); // Output: 2
        System.out.println(stackUsingHeap.pop()); // Output: 1

        /*
         * Q45.
         * Given a string, determine whether any permutation of it is a palindrome.
         * For example, carrace should return true, since it can be rearranged to form
         * racecar, which is a palindrome. daily should return false, since there's no
         * rearrangement that can form a palindrome.
         */
        System.out.println("========= Q45 =========");
        String palindromeString = "carrace";
        String nonPalindromeString = "daily";

        System.out.println(isPermutationPalindrome(palindromeString)); // Output: true
        System.out.println(isPermutationPalindrome(nonPalindromeString)); // Output: false

        /*
         * Q46.
         * Given a string, return the first recurring character in it, or null if there
         * is no recurring character.
         * For example, given the string "acbbac", return "b". Given the string
         * "abcdef", return null.
         */
        System.out.println("========= Q46 =========");
        String strToFindRecurringString1 = "acbbac";
        String strToFindRecurringString2 = "abcdef";

        System.out.println(findFirstRecurringCharacter(strToFindRecurringString1)); // Output: b
        System.out.println(findFirstRecurringCharacter(strToFindRecurringString2)); // Output: null

        /*
         * Q47.
         * Given a 32-bit integer, return the number with its bits reversed.
         * For example, given the binary number 1111 0000 1111 0000 1111 0000 1111 0000,
         * return 0000 1111 0000 1111 0000 1111 0000 1111.
         */
        System.out.println("========= Q47 =========");
        String binary = "11110000111100001111000011110000";
        System.out.println(reverseBinary(binary)); // Output: 00001111000011110000111100001111

        /*
         * Q48.
         * You are given a list of data entries that represent entries and exits of
         * groups of people into a building. An entry looks like this:
         * {"timestamp": 1526579928, count: 3, "type": "enter"}
         * This means 3 people entered the building. An exit looks like this:
         * {"timestamp": 1526580382, count: 2, "type": "exit"}
         * This means that 2 people exited the building. timestamp is in Unix time.
         * Find the busiest period in the building, that is, the time with the most
         * people in the building. Return it as a pair of (start, end) timestamps. You
         * can assume the building always starts off and ends up empty, i.e. with 0
         * people inside.
         */
        System.out.println("========= Q48 =========");
        int[][] data = {
                { 1526579928, 3, 1 }, // Entry: 3 people entered the building
                { 1526579935, 2, 1 }, // Entry: 2 people entered the building
                { 1526579940, 1, 0 }, // Exit: 1 person exited the building
                { 1526579945, 4, 1 }, // Entry: 4 people entered the building
                { 1526579950, 2, 0 }, // Exit: 2 people exited the building
                { 1526579955, 1, 0 }, // Exit: 1 person exited the building
                { 1526579960, 3, 0 } // Exit: 3 people exited the building
        };

        int[] busiestPeriod = findBusiestPeriod(data);
        System.out.println("Busiest Period: [" + busiestPeriod[0] + ", " + busiestPeriod[1] + "]");

        /*
         * Q49.
         * Write a function to flatten a nested dictionary. Namespace the keys with a
         * period.
         * For example, given the following dictionary:
         * "{                      "
         * "    "key": 3,          "
         * "    "foo": {           "
         * "        "a": 5,        "
         * "        "bar": {       "
         * "            "baz": 8   "
         * "        }              "
         * "    }                  "
         * "}                      "
         * it should become:
         * "{                      "
         * "    "key": 3,          "
         * "    "foo.a": 5,        "
         * "    "foo.bar.baz": 8   "
         * "}                      "
         * You can assume keys do not contain dots in them, i.e. no clobbering will
         * occur.
         */
        System.out.println("========= Q49 =========");
        Map<String, Object> dict = new HashMap<>();
        dict.put("key", 3);
        Map<String, Object> foo = new HashMap<>();
        foo.put("a", 5);
        Map<String, Object> bar = new HashMap<>();
        bar.put("baz", 8);
        foo.put("bar", bar);
        dict.put("foo", foo);

        Map<String, Object> flattenedDict = flattenDictionary(dict);
        System.out.println(flattenedDict);

        /*
         * Q50.
         * You are given a starting state start, a list of transition probabilities for
         * a Markov chain, and a number of steps num_steps. Run the Markov chain
         * starting from start for num_steps and compute the number of times we visited
         * each state.
         * For example, given the starting state a, number of steps 5000, and the
         * following transition probabilities:
         * "[                      "
         * "  ('a', 'a', 0.9),     "
         * "  ('a', 'b', 0.075),   "
         * "  ('a', 'c', 0.025),   "
         * "  ('b', 'a', 0.15),    "
         * "  ('b', 'b', 0.8),     "
         * "  ('b', 'c', 0.05),    "
         * "  ('c', 'a', 0.25),    "
         * "  ('c', 'b', 0.25),    "
         * "  ('c', 'c', 0.5)      "
         * "]                      "
         * One instance of running this Markov chain might produce { 'a': 3012, 'b':
         * 1656, 'c': 332 }.
         */
        System.out.println("========= Q50 =========");
        char startState = 'a';
        int numSteps = 5000;
        List<TransitionProbability> transitionProbabilities = new ArrayList<>();
        transitionProbabilities.add(new TransitionProbability('a', 'a', 0.9));
        transitionProbabilities.add(new TransitionProbability('a', 'b', 0.075));
        transitionProbabilities.add(new TransitionProbability('a', 'c', 0.025));
        transitionProbabilities.add(new TransitionProbability('b', 'a', 0.15));
        transitionProbabilities.add(new TransitionProbability('b', 'b', 0.8));
        transitionProbabilities.add(new TransitionProbability('b', 'c', 0.05));
        transitionProbabilities.add(new TransitionProbability('c', 'a', 0.25));
        transitionProbabilities.add(new TransitionProbability('c', 'b', 0.25));
        transitionProbabilities.add(new TransitionProbability('c', 'c', 0.5));

        Map<Character, Integer> stateCounts = simulateMarkovChain(startState, numSteps, transitionProbabilities);
        System.out.println(stateCounts);

        /*
         * Q51.
         * Determine whether there exists a one-to-one character mapping from one string
         * s1 to another s2.
         * For example, given s1 = abc and s2 = bcd, return true since we can map a to
         * b, b to c, and c to d.
         * Given s1 = foo and s2 = bar, return false since the o cannot map to two
         * characters.
         */
        System.out.println("========= Q51 =========");
        String s1 = "abc";
        String s2 = "bcd";
        System.out.println(isCharacterMapping(s1, s2)); // Output: true

        s1 = "foo";
        s2 = "bar";
        System.out.println(isCharacterMapping(s1, s2)); // Output: false

        /*
         * Q52.
         * Given a linked list and a positive integer k, rotate the list to the right by
         * k places.
         * For example, given the linked list 7 -> 7 -> 3 -> 5 and k = 2, it should
         * become 3 -> 5 -> 7 -> 7.
         * Given the linked list 1 -> 2 -> 3 -> 4 -> 5 and k = 3, it should become 3 ->
         * 4 -> 5 -> 1 -> 2.
         */
        System.out.println("========= Q52 =========");
        ListNode head1 = new ListNode(7);
        head1.next = new ListNode(7);
        head1.next.next = new ListNode(3);
        head1.next.next.next = new ListNode(5);

        int k1 = 2;
        ListNode rotated1 = rotateRight(head1, k1);
        printLinkedList(rotated1); // Output: 3 -> 5 -> 7 -> 7

        ListNode head2 = new ListNode(1);
        head2.next = new ListNode(2);
        head2.next.next = new ListNode(3);
        head2.next.next.next = new ListNode(4);
        head2.next.next.next.next = new ListNode(5);

        int k2 = 3;
        ListNode rotated2 = rotateRight(head2, k2);
        printLinkedList(rotated2); // Output: 3 -> 4 -> 5 -> 1 -> 2

        /*
         * Q53.
         * Given n numbers, find the greatest common denominator between them.
         * For example, given the numbers [42, 56, 14], return 14.
         */
        System.out.println("========= Q53 =========");
        int[] numbers = { 42, 56, 14 };

        BigInteger gcd = greatestCommonDenominator(numbers);
        System.out.print("Greatest Common Denominator: ");
        System.out.println(gcd);

        /*
         * Q54.
         * Given two rectangles on a 2D graph, return the area of their intersection. If
         * the rectangles don't intersect, return 0.
         * For example, given the following rectangles:
         * "{                                          "
         * "    "top_left": (1, 4),                    "
         * "    "dimensions": (3, 3) # width, height   "
         * "}                                          "
         * and
         * "{                                          "
         * "    "top_left": (0, 5),                    "
         * "    "dimensions": (4, 3) # width, height   "
         * "}                                          "
         * return 6.
         */
        System.out.println("========= Q54 =========");
        Rectangle rectangle1 = new Rectangle(1, 4, 3, 3);
        Rectangle rectangle2 = new Rectangle(0, 5, 4, 3);

        int intersectionArea = rectangle1.getIntersectionArea(rectangle2);

        System.out.println("Intersection Area: " + intersectionArea);

        /*
         * Q55.
         * You are given given a list of rectangles represented by min and max x- and
         * y-coordinates. Compute whether or not a pair of rectangles overlap each
         * other. If one rectangle completely covers another, it is considered
         * overlapping.
         * For example, given the following rectangles:
         * "{                                          "
         * "    "top_left": (1, 4),                    "
         * "    "dimensions": (3, 3) # width, height   "
         * "},                                         "
         * "{                                          "
         * "    "top_left": (-1, 3),                   "
         * "    "dimensions": (2, 1)                   "
         * "},                                         "
         * "{                                          "
         * "    "top_left": (0, 5),                    "
         * "    "dimensions": (4, 3)                   "
         * "}                                          "
         * return true as the first and third rectangle overlap each other.
         */
        System.out.println("========= Q55 =========");
        Rectangle[] rectangles = new Rectangle[] {
                new Rectangle(1, 4, 3, 3),
                new Rectangle(-1, 3, 2, 1),
                new Rectangle(0, 5, 4, 3)
        };

        boolean overlapExists = checkOverlap(rectangles);

        System.out.println("Overlap Exists: " + overlapExists);

        /*
         * Q56.
         * Given an array of elements, return the length of the longest subarray where
         * all its elements are distinct.
         * For example, given the array [5, 1, 3, 5, 2, 3, 4, 1], return 5 as the
         * longest subarray of distinct elements is [5, 2, 3, 4, 1].
         */
        System.out.println("========= Q56 =========");
        int[] arrToFindLongestUniqueSubarray = { 5, 1, 3, 5, 2, 3, 4, 1 };
        int longestSubarrayLength = findLongestSubarrayLength(arrToFindLongestUniqueSubarray);

        System.out.println("Longest Subarray Length: " + longestSubarrayLength);

        /*
         * Q57.
         * Given a collection of intervals, find the minimum number of intervals you
         * need to remove to make the rest of the intervals non-overlapping.
         * Intervals can "touch", such as [0, 1] and [1, 2], but they won't be
         * considered overlapping.
         * For example, given the intervals (7, 9), (2, 4), (5, 8), return 1 as the last
         * interval can be removed and the first two won't overlap.
         * The intervals are not necessarily sorted in any order.
         */
        System.out.println("========= Q57 =========");
        int[][] collectionOfIntervals = { { 7, 9 }, { 2, 4 }, { 5, 8 } };
        int minIntervalsToRemove = eraseOverlapIntervals(collectionOfIntervals);

        System.out.println("Minimum Intervals to Remove: " + minIntervalsToRemove);

        /*
         * Q58.
         * Suppose you are given two lists of n points, one list p1, p2, ..., pn on the
         * line y = 0 and the other list q1, q2, ..., qn on the line y = 1. Imagine a
         * set of n line segments connecting each point pi to qi. Write an algorithm to
         * determine how many pairs of the line segments intersect.
         */
        System.out.println("========= Q58 =========");
        int[] p = { 1, 2, 3, 4 };
        int[] q = { 5, 6, 7, 8 };
        int intersectCount = countIntersectingPairs(p, q);

        System.out.println("Number of Intersecting Pairs: " + intersectCount);

        /*
         * Q59.
         * Given the root of a binary tree, find the most frequent subtree sum. The
         * subtree sum of a node is the sum of all values under a node, including the
         * node itself.
         * For example, given the following tree:
         * "  5    "
         * " / \   "
         * "2  -5  "
         * Return 2 as it occurs twice: once as the left leaf, and once as the sum of 2
         * + 5 - 5.
         */
        System.out.println("========= Q59 =========");
        TreeNode<Integer> mostSubtreeSum = new TreeNode<>(5);
        mostSubtreeSum.left = new TreeNode<>(2);
        mostSubtreeSum.right = new TreeNode<>(-5);

        int mostFrequentSum = findMostFrequentSubtreeSum(mostSubtreeSum);
        System.out.println("Most frequent subtree sum: " + mostFrequentSum);

        /*
         * Q60.
         * Given an array and a number k that's smaller than the length of the array,
         * rotate the array to the right k elements in-place.
         */
        System.out.println("========= Q60 =========");
        int[] arrayToRotate = { 1, 2, 3, 4, 5 };
        int numOfRotation = 3;

        rotate(arrayToRotate, numOfRotation);

        System.out.println("Rotated array:");
        for (int numToRotate : arrayToRotate) {
            System.out.print(numToRotate + " ");
        }

        /*
         * Q61.
         * You are given an array of arrays of integers, where each array corresponds to
         * a row in a triangle of numbers. For example, [[1], [2, 3], [1, 5, 1]]
         * represents the triangle:
         * "  1    "
         * " 2 3   "
         * "1 5 1  "
         * We define a path in the triangle to start at the top and go down one row at a
         * time to an adjacent value, eventually ending with an entry on the bottom row.
         * For example, 1 -> 3 -> 5. The weight of the path is the sum of the entries.
         * Write a program that returns the weight of the maximum weight path.
         */
        System.out.println("========= Q61 =========");
        int[][] triangle = {
                { 1 },
                { 2, 3 },
                { 1, 5, 1 }
        };

        int maxPathSum = maximumPathSum(triangle);
        System.out.println("Maximum Path Sum: " + maxPathSum);

        /*
         * Q62.
         * Write a program that checks whether an integer is a palindrome. For example,
         * 121 is a palindrome, as well as 888. 678 is not a palindrome. Do not convert
         * the integer into a string.
         */
        System.out.println("========= Q62 =========");
        int number1 = 121;
        int number2 = 888;
        int number3 = 678;

        System.out.println(number1 + " is a palindrome: " + isPalindrome(number1));
        System.out.println(number2 + " is a palindrome: " + isPalindrome(number2));
        System.out.println(number3 + " is a palindrome: " + isPalindrome(number3));

        /*
         * Q63.
         * Given a complete binary tree, count the number of nodes in faster than O(n)
         * time. Recall that a complete binary tree has every level filled except the
         * last, and the nodes in the last level are filled starting from the left.
         */
        System.out.println("========= Q63 =========");
        TreeNode<Integer> completeBT = new TreeNode<>(1);
        completeBT.left = new TreeNode<>(2);
        completeBT.right = new TreeNode<>(3);
        completeBT.left.left = new TreeNode<>(4);
        completeBT.left.right = new TreeNode<>(5);
        completeBT.right.left = new TreeNode<>(6);

        System.out.print("Is this binary tree complete? (true/false): ");
        System.out.println(isCompleteBinaryTree(completeBT));

        /*
         * Q64.
         * Given an integer, find the next permutation of it in absolute order. For
         * example, given 48975, the next permutation would be 49578.
         */
        System.out.println("========= Q64 =========");
        int numToFindNextPermutation = 48975;
        int nextPermutation = findNextPermutation(numToFindNextPermutation);
        System.out.println("Next permutation: " + nextPermutation);

        /*
         * Q65.
         * A permutation can be specified by an array P, where P[i] represents the
         * location of the element at i in the permutation. For example, [2, 1, 0]
         * represents the permutation where elements at the index 0 and 2 are swapped.
         * Given an array and a permutation, apply the permutation to the array. For
         * example, given the array ["a", "b", "c"] and the permutation [2, 1, 0],
         * return ["c", "b", "a"].
         */
        System.out.println("========= Q65 =========");
        String[] array = { "a", "b", "c" };
        int[] permutation = { 2, 1, 0 };
        String[] resultFromPermutation = applyPermutation(array, permutation);
        System.out.println(Arrays.toString(resultFromPermutation));

        /*
         * Q66.
         * A Collatz sequence in mathematics can be defined as follows. Starting with
         * any positive integer:
         * if n is even, the next number in the sequence is n / 2
         * if n is odd, the next number in the sequence is 3n + 1
         * It is conjectured that every such sequence eventually reaches the number 1.
         * Test this conjecture.
         * Bonus: What input n <= 1000000 gives the longest sequence?
         */
        System.out.println("========= Q66 =========");
        long longestSequence = 0;
        long longestSequenceNumber = 0;

        for (long i = 1; i <= 1000000; i++) {
            long sequenceLength = collatzSequence(i);
            if (sequenceLength > longestSequence) {
                longestSequence = sequenceLength;
                longestSequenceNumber = i;
            }
        }

        System.out.println("Longest sequence: " + longestSequence);
        System.out.println("Input n: " + longestSequenceNumber);

        /*
         * Q67.
         * Spreadsheets often use this alphabetical encoding for its columns: "A", "B",
         * "C", ..., "AA", "AB", ..., "ZZ", "AAA", "AAB", ....
         * Given a column number, return its alphabetical column id. For example, given
         * 1, return "A". Given 27, return "AA".
         */
        System.out.println("========= Q67 =========");
        int columnNumber1 = 1;
        int columnNumber2 = 27;

        String columnID1 = getColumnID(columnNumber1);
        String columnID2 = getColumnID(columnNumber2);

        System.out.println("Column ID for " + columnNumber1 + ": " + columnID1);
        System.out.println("Column ID for " + columnNumber2 + ": " + columnID2);

        /*
         * Q68.
         * Given an integer n, return the length of the longest consecutive run of 1s in
         * its binary representation.
         * For example, given 156, you should return 3.
         */
        System.out.println("========= Q68 =========");
        int numToFindLongest1s = 156;
        int longestRun = longestConsecutiveRun(numToFindLongest1s);
        System.out.println("Longest consecutive run of 1s in " + n + ": " + longestRun);

        /*
         * Q69.
         * Let's define a "sevenish" number to be one which is either a power of 7, or
         * the sum of unique powers of 7. The first few sevenish numbers are 1, 7, 8,
         * 49, and so on. Create an algorithm to find the nth sevenish number.
         */
        System.out.println("========= Q69 =========");
        int nthSevenish = 5;
        int nthSevenishNumber = getNthSevenishNumber(nthSevenish);
        System.out.println("The " + nthSevenish + "th sevenish number is: " + nthSevenishNumber);

        /*
         * Q70.
         * Given a sorted array, find the smallest positive integer that is not the sum
         * of a subset of the array.
         * For example, for the input [1, 2, 3, 10], you should return 7.
         * Do this in O(N) time.
         */
        System.out.println("========= Q70 =========");
        int[] numsToFindSmallestInteger = { 1, 2, 3, 10 };
        int smallestInteger = findSmallestPositiveInteger(numsToFindSmallestInteger);
        System.out.println("Smallest positive integer: " + smallestInteger);

        /*
         * Q71.
         * There are N prisoners standing in a circle, waiting to be executed. The
         * executions are carried out starting with the kth person, and removing every
         * successive kth person going clockwise until there is no one left.
         * Given N and k, write an algorithm to determine where a prisoner should stand
         * in order to be the last survivor.
         * For example, if N = 5 and k = 2, the order of executions would be [2, 4, 1,
         * 5, 3], so you should return 3.
         * Bonus: Find an O(log N) solution if k = 2.
         */
        System.out.println("========= Q71 =========");
        int nPrisoners = 5;
        int kthPrisoner = 2;
        int lastSurvivor = findLastPrisoner(nPrisoners, kthPrisoner);
        System.out.println("The last survivor's position is: " + lastSurvivor);

        /*
         * Q72.
         * Boggle is a game played on a 4 x 4 grid of letters. The goal is to find as
         * many words as possible that can be formed by a sequence of adjacent letters
         * in the grid, using each cell at most once. Given a game board and a
         * dictionary of valid words, implement a Boggle solver.
         */
        System.out.println("========= Q72 =========");
        char[][] boggleBoard = {
                { 'A', 'B', 'C', 'D' },
                { 'E', 'F', 'G', 'H' },
                { 'I', 'J', 'K', 'L' },
                { 'M', 'N', 'O', 'P' }
        };
        String[] dictionary = { "ABEF", "BFJ", "CDEG", "PONM", "PONMLKJIHGFEDCBA" };
        Set<String> foundWords = findWords(boggleBoard, dictionary);
        System.out.println("Found words: " + foundWords);

        /*
         * Q73.
         * Given a string with repeated characters, rearrange the string so that no two
         * adjacent characters are the same. If this is not possible, return None.
         * For example, given "aaabbc", you could return "ababac". Given "aaab", return
         * None.
         */
        System.out.println("========= Q73 =========");
        String input1 = "aaabbc";
        String rearranged1 = rearrangeString(input1);
        System.out.println("Rearranged string for " + input1 + ": " + rearranged1);

        String input2 = "aaab";
        String rearranged2 = rearrangeString(input2);
        System.out.println("Rearranged string for " + input2 + ": " + rearranged2);

        /*
         * Q74.
         * Implement a PrefixMapSum class with the following methods:
         * insert(key: str, value: int): Set a given key's value in the map. If the key
         * already exists, overwrite the value.
         * sum(prefix: str): Return the sum of all values of keys that begin with a
         * given prefix.
         * For example, you should be able to run the following code:
         * mapsum.insert("columnar", 3)
         * assert mapsum.sum("col") == 3
         * 
         * mapsum.insert("column", 2)
         * assert mapsum.sum("col") == 5
         */
        System.out.println("========= Q74 =========");
        PrefixMapSum mapsum = new PrefixMapSum();

        mapsum.insert("columnar", 3);
        System.out.println(mapsum.sum("col") == 3);

        mapsum.insert("column", 2);
        System.out.println(mapsum.sum("col") == 5);

        /*
         * Q75.
         * Implement the function fib(n), which returns the nth number in the Fibonacci
         * sequence, using only O(1) space.
         */
        System.out.println("========= Q75 =========");
        int nthFib = 6;
        int nthFibonacciNumber = fib(n);
        System.out.println("Fibonacci number at index " + nthFib + ": " + nthFibonacciNumber);

        /*
         * Q76.
         * A tree is symmetric if its data and shape remain unchanged when it is
         * reflected about the root node. The following tree is an example:
         * "        4          "
         * "      / | \        "
         * "    3   5   3      "
         * "  /           \    "
         * "9              9   "
         * Given a k-ary tree, determine whether it is symmetric.
         */
        System.out.println("========= Q76 =========");
        NonBinaryTreeNode symmetricTree = new NonBinaryTreeNode(4);
        NonBinaryTreeNode node1 = new NonBinaryTreeNode(3);
        NonBinaryTreeNode node2 = new NonBinaryTreeNode(5);
        NonBinaryTreeNode node3 = new NonBinaryTreeNode(3);
        NonBinaryTreeNode node4 = new NonBinaryTreeNode(9);
        NonBinaryTreeNode node5 = new NonBinaryTreeNode(9);

        symmetricTree.children.add(node1);
        symmetricTree.children.add(node2);
        symmetricTree.children.add(node3);
        node1.children.add(node4);
        node3.children.add(node5);

        boolean isSymmetric = isSymmetric(symmetricTree);
        System.out.println("Is the tree symmetric? " + isSymmetric);

        /*
         * Q77.
         * In academia, the h-index is a metric used to calculate the impact of a
         * researcher's papers. It is calculated as follows:
         * A researcher has index h if at least h of her N papers have h citations each.
         * If there are multiple h satisfying this formula, the maximum is chosen.
         * For example, suppose N = 5, and the respective citations of each paper are
         * [4, 3, 0, 1, 5]. Then the h-index would be 3, since the researcher has 3
         * papers with at least 3 citations.
         * Given a list of paper citations of a researcher, calculate their h-index.
         */
        System.out.println("========= Q77 =========");
        int[] citations = { 4, 3, 0, 1, 5 };
        int hIndex = calculateHIndex(citations);
        System.out.println("The h-index is: " + hIndex);

        /*
         * Q78.
         * The Sieve of Eratosthenes is an algorithm used to generate all prime numbers
         * smaller than N. The method is to take increasingly larger prime numbers, and
         * mark their multiples as composite.
         * For example, to find all primes less than 100, we would first mark [4, 6, 8,
         * ...] (multiples of two), then [6, 9, 12, ...] (multiples of three), and so
         * on. Once we have done this for all primes less than N, the unmarked numbers
         * that remain will be prime.
         * Implement this algorithm.
         * Bonus: Create a generator that produces primes indefinitely (that is, without
         * taking N as an input).
         */
        System.out.println("========= Q78 =========");
        int nPrime = 100;
        List<Integer> primeNums = generatePrimes(nPrime);
        System.out.println("Prime numbers less than " + nPrime + ": " + primeNums);

        PrimeGenerator primeGenerator = new PrimeGenerator();

        Iterator<Integer> iterator = primeGenerator.iterator();
        for (int i = 0; i < nPrime; i++) {
            System.out.println(iterator.next());
        }

        /*
         * Q79.
         * Given a binary tree, determine whether or not it is height-balanced. A
         * height-balanced binary tree can be defined as one in which the heights of the
         * two subtrees of any node never differ by more than one.
         */
        System.out.println("========= Q79 =========");
        TreeNode<Integer> balancedBinaryTree = new TreeNode<>(1);
        balancedBinaryTree.left = new TreeNode<>(2);
        balancedBinaryTree.right = new TreeNode<>(3);
        balancedBinaryTree.left.left = new TreeNode<>(4);
        balancedBinaryTree.left.right = new TreeNode<>(5);
        balancedBinaryTree.left.left.left = new TreeNode<>(6);

        BalancedBinaryTree bbt = new BalancedBinaryTree();
        boolean isBalanced = bbt.isBalanced(balancedBinaryTree);
        System.out.println("Is the binary tree balanced? " + isBalanced);

        /*
         * Q80.
         * The ancient Egyptians used to express fractions as a sum of several terms
         * where each numerator is one. For example, 4 / 13 can be represented as 1 / 4
         * + 1 / 18 + 1 / 468.
         * Create an algorithm to turn an ordinary fraction a / b, where a < b, into an
         * Egyptian fraction.
         */
        System.out.println("========= Q80 =========");
        int numerator = 4;
        int denominator = 13;

        List<String> egyptianFractions = convertToEgyptianFraction(numerator, denominator);

        System.out.printf("Egyptian Fraction representation of %d/%d: ", numerator, denominator);
        for (int i = 0; i < egyptianFractions.size(); i++) {
            System.out.print(egyptianFractions.get(i));
            if (i < egyptianFractions.size() - 1) {
                System.out.print(" + ");
            }
        }

        /*
         * Q81.
         * The transitive closure of a graph is a measure of which vertices are
         * reachable from other vertices. It can be represented as a matrix M, where
         * M[i][j] == 1 if there is a path between vertices i and j, and otherwise 0.
         * For example, suppose we are given the following graph in adjacency list form:
         * graph = [
         * [0, 1, 3],
         * [1, 2],
         * [2],
         * [3]
         * ]
         * The transitive closure of this graph would be:
         * [1, 1, 1, 1]
         * [0, 1, 1, 0]
         * [0, 0, 1, 0]
         * [0, 0, 0, 1]
         * Given a graph, find its transitive closure.
         */
        System.out.println("========= Q81 =========");
        int[][] graph = {
                { 0, 1, 3 },
                { 1, 2 },
                { 2 },
                { 3 }
        };

        int[][] closure = findTransitiveClosure(graph);

        System.out.println("Transitive Closure:");
        for (int i = 0; i < closure.length; i++) {
            System.out.println(Arrays.toString(closure[i]));
        }

        /*
         * Q82.
         * Given an array of integers out of order, determine the bounds of the smallest
         * window that must be sorted in order for the entire array to be sorted. For
         * example, given [3, 7, 5, 6, 9], you should return (1, 3).
         */
        System.out.println("========= Q82 =========");
        int[] arr = { 3, 7, 5, 6, 9 };
        int[] bounds = findBounds(arr);
        System.out.println(Arrays.toString(bounds)); // Output: [1, 3]

        /*
         * Q83.
         * In Ancient Greece, it was common to write text with the first line going left
         * to right, the second line going right to left, and continuing to go back and
         * forth. This style was called "boustrophedon".
         * Given a binary tree, write an algorithm to print the nodes in boustrophedon
         * order.
         * For example, given the following tree:
         * "       1           "
         * "    /     \        "
         * "  2         3      "
         * " / \       / \     "
         * "4   5     6   7    "
         * You should return [1, 3, 2, 4, 5, 6, 7].
         */
        System.out.println("========= Q83 =========");
        TreeNode<Integer> boustrophedonTree = new TreeNode<>(1);
        boustrophedonTree.left = new TreeNode<>(2);
        boustrophedonTree.right = new TreeNode<>(3);
        boustrophedonTree.left.left = new TreeNode<>(4);
        boustrophedonTree.left.right = new TreeNode<>(5);
        boustrophedonTree.right.left = new TreeNode<>(6);
        boustrophedonTree.right.right = new TreeNode<>(7);

        printBoustrophedon(boustrophedonTree);

        /*
         * Q84.
         * Huffman coding is a method of encoding characters based on their frequency.
         * Each letter is assigned a variable-length binary string, such as 0101 or
         * 111110, where shorter lengths correspond to more common letters. To
         * accomplish this, a binary tree is built such that the path from the root to
         * any leaf uniquely maps to a character. When traversing the path, descending
         * to a left child corresponds to a 0 in the prefix, while descending right
         * corresponds to 1.
         * Here is an example tree (note that only the leaf nodes have letters):
         * 
         * "        *          "
         * "      /   \        "
         * "    *       *      "
         * "   / \     / \     "
         * "  *   a   t   *    "
         * " /             \   "
         * "c               s  "
         * With this encoding, cats would be represented as 0000110111.
         * Given a dictionary of character frequencies, build a Huffman tree, and use it
         * to determine a mapping between characters and their encoded binary strings.
         */
        System.out.println("========= Q84 =========");
        TreeNode<Character> huffmanTreeRoot = new TreeNode<>('*');
        huffmanTreeRoot.left = new TreeNode<>('*');
        huffmanTreeRoot.right = new TreeNode<>('*');
        huffmanTreeRoot.left.left = new TreeNode<>('*');
        huffmanTreeRoot.left.right = new TreeNode<>('a');
        huffmanTreeRoot.right.left = new TreeNode<>('t');
        huffmanTreeRoot.right.right = new TreeNode<>('*');
        huffmanTreeRoot.left.left.left = new TreeNode<>('c');
        huffmanTreeRoot.right.right.right = new TreeNode<>('s');

        String huffmanCoding = buildHuffmanTree(huffmanTreeRoot);
        System.out.println(huffmanCoding); // Output: 0000110111

        /*
         * Q85.
         * MegaCorp wants to give bonuses to its employees based on how many lines of
         * codes they have written. They would like to give the smallest positive amount
         * to each worker consistent with the constraint that if a developer has written
         * more lines of code than their neighbor, they should receive more money.
         * Given an array representing a line of seats of employees at MegaCorp,
         * determine how much each one should get paid.
         * For example, given [10, 40, 200, 1000, 60, 30], you should return [1, 2, 3,
         * 4, 2, 1].
         */
        System.out.println("========= Q85 =========");
        int[] linesOfCode = { 10, 40, 200, 1000, 60, 30 };
        int[] bonuses = calculateBonuses(linesOfCode);

        // Print the calculated bonuses
        for (int bonus : bonuses) {
            System.out.print(bonus + " ");
        }

        /*
         * Q86.
         * A step word is formed by taking a given word, adding a letter, and
         * anagramming the result. For example, starting with the word "APPLE", you can
         * add an "A" and anagram to get "APPEAL".
         * Given a dictionary of words and an input word, create a function that returns
         * all valid step words.
         */
        System.out.println("========= Q86 =========");
        String inputWord = "APPLE";
        List<String> dictionaryForAnagram = Arrays.asList("APPEAL", "PEAR", "PLEA", "LEAP");
        List<String> stepWords = findStepWords(inputWord, dictionaryForAnagram);
        System.out.println(stepWords);

        /*
         * Q87.
         * You are given an string representing the initial conditions of some dominoes.
         * Each element can take one of three values:
         * L, meaning the domino has just been pushed to the left,
         * R, meaning the domino has just been pushed to the right, or
         * ., meaning the domino is standing still.
         * Determine the orientation of each tile when the dominoes stop falling. Note
         * that if a domino receives a force from the left and right side
         * simultaneously, it will remain upright.
         * For example, given the string .L.R....L, you should return LL.RRRLLL.
         * Given the string ..R...L.L, you should return ..RR.LLLL.
         */
        System.out.println("========= Q87 =========");
        String dominoes1 = ".L.R....L";
        String dominoes2 = "..R...L.L";

        String oriented1 = orientDominos(dominoes1);
        String oriented2 = orientDominos(dominoes2);

        System.out.println(oriented1); // Output: LL.RRRLLL
        System.out.println(oriented2); // Output: ..RR.LLLL

        /*
         * Q88.
         * A fixed point in an array is an element whose value is equal to its index.
         * Given a sorted array of distinct elements, return a fixed point, if one
         * exists. Otherwise, return False.
         * For example, given [-6, 0, 2, 40], you should return 2. Given [1, 5, 7, 8],
         * you should return False.
         */
        System.out.println("========= Q88 =========");
        int[] nums1 = { -6, 0, 2, 40 };
        int[] nums2 = { 1, 5, 7, 8 };

        int fixedPoint1 = findFixedPoint(nums1);
        int fixedPoint2 = findFixedPoint(nums2);

        System.out.println(fixedPoint1); // Output: 2
        System.out.println(fixedPoint2); // Output: -1 (False)

        /*
         * Q89.
         * UTF-8 is a character encoding that maps each symbol to one, two, three, or
         * four bytes.
         * For example, the Euro sign, €, corresponds to the three bytes 11100010
         * 10000010 10101100. The rules for mapping characters are as follows:
         * For a single-byte character, the first bit must be zero.
         * For an n-byte character, the first byte starts with n ones and a zero. The
         * other n - 1 bytes all start with 10.
         * Visually, this can be represented as follows.
         * Bytes | Byte format
         * -----------------------------------------------
         * 1 | 0xxxxxxx
         * 2 | 110xxxxx 10xxxxxx
         * 3 | 1110xxxx 10xxxxxx 10xxxxxx
         * 4 | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
         * Write a program that takes in an array of integers representing byte values,
         * and returns whether it is a valid UTF-8 encoding.
         */
        System.out.println("========= Q89 =========");
        int[] data1 = { 197, 130, 1 }; // Valid UTF-8 encoding
        int[] data2 = { 235, 140, 4 }; // Invalid UTF-8 encoding

        System.out.println(validUtf8(data1)); // Output: true
        System.out.println(validUtf8(data2)); // Output: false

        /*
         * Q90.
         * Given an integer N, construct all possible binary search trees with N nodes.
         */
        System.out.println("========= Q90 =========");
        int nForBinarySearchTree = 3;
        List<TreeNode<Integer>> possibleBSTs = generateBSTs(nForBinarySearchTree);

        for (TreeNode<Integer> tree : possibleBSTs) {
            printTree(tree);
            System.out.println();
        }

        /*
         * Q91.
         * A classroom consists of N students, whose friendships can be represented in
         * an adjacency list. For example, the following describes a situation where 0
         * is friends with 1 and 2, 3 is friends with 6, and so on.
         * {0: [1, 2],
         * 1: [0, 5],
         * 2: [0],
         * 3: [6],
         * 4: [],
         * 5: [1],
         * 6: [3]}
         * Each student can be placed in a friend group, which can be defined as the
         * transitive closure of that student's friendship relations. In other words,
         * this is the smallest set such that no student in the group has any friends
         * outside this group. For the example above, the friend groups would be {0, 1,
         * 2, 5}, {3, 6}, {4}.
         * Given a friendship list such as the one above, determine the number of friend
         * groups in the class.
         */
        System.out.println("========= Q91 =========");
        int[][] adjacencyList = {
                { 1, 2 },
                { 0, 5 },
                { 0 },
                { 6 },
                {},
                { 1 },
                { 3 }
        };

        int friendGroups = countFriendGroups(adjacencyList);
        System.out.println("Number of Friend Groups: " + friendGroups);

        /*
         * Q92.
         * Given an undirected graph, determine if it contains a cycle.
         */
        System.out.println("========= Q92 =========");
        int vertices = 5;
        Graph graphWithCycle = new Graph(vertices);
        graphWithCycle.addEdge(0, 1);
        graphWithCycle.addEdge(1, 2);
        graphWithCycle.addEdge(2, 3);
        graphWithCycle.addEdge(3, 4);
        graphWithCycle.addEdge(4, 1);

        boolean hasCycle = graphWithCycle.containsCycle();
        System.out.println("Graph contains a cycle: " + hasCycle);

        /*
         * Q93.
         * Given an array of integers, determine whether it contains a Pythagorean
         * triplet. Recall that a Pythagorean triplet (a, b, c) is defined by the
         * equation a2+ b2= c2.
         */
        System.out.println("========= Q93 =========");
        int[] numsForPythagorean = { 3, 1, 4, 6, 5 };

        boolean hasPythagoreanTriplet = containsPythagoreanTriplet(numsForPythagorean);
        System.out.println("Array contains a Pythagorean triplet: " + hasPythagoreanTriplet);

        /*
         * Q94.
         * A regular number in mathematics is defined as one which evenly divides some
         * power of 60. Equivalently, we can say that a regular number is one whose only
         * prime divisors are 2, 3, and 5.
         * These numbers have had many applications, from helping ancient Babylonians
         * keep time to tuning instruments according to the diatonic scale.
         * Given an integer N, write a program that returns, in order, the first N
         * regular numbers.
         */
        System.out.println("========= Q94 =========");
        int nForRegularNums = 10;

        List<Long> regularNumbers = getRegularNumbers(nForRegularNums);
        System.out.println("First " + n + " regular numbers:");
        for (long regularNum : regularNumbers) {
            System.out.print(regularNum + " ");
        }

        /*
         * Q95.
         * On a mysterious island there are creatures known as Quxes which come in three
         * colors: red, green, and blue. One power of the Qux is that if two of them are
         * standing next to each other, they can transform into a single creature of the
         * third color.
         * Given N Quxes standing in a line, determine the smallest number of them
         * remaining after any possible sequence of such transformations.
         * For example, given the input ['R', 'G', 'B', 'G', 'B'], it is possible to end
         * up with a single Qux through the following steps:
         * "        Arrangement       |   Change    "
         * "----------------------------------------"
         * "['R', 'G', 'B', 'G', 'B'] | (R, G) -> B "
         * "['B', 'B', 'G', 'B']      | (B, G) -> R "
         * "['B', 'R', 'B']           | (R, B) -> G "
         * "['B', 'G']                | (B, G) -> R "
         * "['R']                     |             "
         */
        System.out.println("========= Q95 =========");
        char[] quxes = { 'R', 'G', 'B', 'G', 'B' };
        int remainingQuxes = getRemainingQuxes(quxes);
        System.out.println("Smallest number of remaining Quxes: " + remainingQuxes);

        /*
         * Q96.
         * A girl is walking along an apple orchard with a bag in each hand. She likes
         * to pick apples from each tree as she goes along, but is meticulous about not
         * putting different kinds of apples in the same bag.
         * Given an input describing the types of apples she will pass on her path, in
         * order, determine the length of the longest portion of her path that consists
         * of just two types of apple trees.
         * For example, given the input [2, 1, 2, 3, 3, 1, 3, 5], the longest portion
         * will involve types 1 and 3, with a length of four.
         */
        System.out.println("========= Q96 =========");
        int[] trees = { 2, 1, 2, 3, 3, 1, 3, 5 };
        int longestPortion = longestTwoAppleTrees(trees);
        System.out.println("Length of the longest portion with two types of apple trees: " + longestPortion);

        /*
         * Q97.
         * On election day, a voting machine writes data in the form (voter_id,
         * candidate_id) to a text file. Write a program that reads this file as a
         * stream and returns the top 3 candidates at any given time. If you find a
         * voter voting more than once, report this as fraud.
         */
        System.out.println("========= Q97 =========");
        // NOT tested
        String filePath = "voting_data.txt"; // Path to the voting data file
        Map<Integer, Integer> voteCounts = new HashMap<>();
        PriorityQueue<Candidate> topCandidates = new PriorityQueue<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] voteData = line.split(",");
                int voterId = Integer.parseInt(voteData[0]);
                int candidateId = Integer.parseInt(voteData[1]);

                if (voteCounts.containsKey(voterId)) {
                    System.out.println("Voter fraud detected: Voter " + voterId + " voted more than once.");
                    continue;
                }

                voteCounts.put(voterId, candidateId);
                int count = voteCounts.getOrDefault(candidateId, 0) + 1;
                voteCounts.put(candidateId, count);

                Candidate candidate = new Candidate(candidateId, count);
                topCandidates.offer(candidate);

                if (topCandidates.size() > 3) {
                    topCandidates.poll();
                }

                System.out.println("Top 3 candidates:");
                for (Candidate c : topCandidates) {
                    System.out.println("Candidate " + c.getId() + ": " + c.getCount() + " votes");
                }
                System.out.println();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        /*
         * Q98.
         * Given a clock time in hh:mm format, determine, to the nearest degree, the
         * angle between the hour and the minute hands.
         * Bonus: When, during the course of a day, will the angle be zero?
         */
        System.out.println("========= Q98 =========");
        String time = "10:45";
        double angle = calculateAngle(time);
        System.out.println("Angle between the hour and minute hands: " + angle + " degrees");

        // Bonus: Find when the angle is zero
        int count = 0;
        for (int hour = 0; hour < 12; hour++) {
            for (int minute = 0; minute < 60; minute++) {
                double currAngle = calculateAngle(hour + ":" + minute);
                if (currAngle == 0) {
                    System.out.println("Angle is zero at " + hour + ":" + minute);
                    count++;
                }
            }
        }
        System.out.println("Total instances where angle is zero: " + count);

        /*
         * Q99.
         * Given a linked list, remove all consecutive nodes that sum to zero. Print out
         * the remaining nodes.
         * For example, suppose you are given the input 3 -> 4 -> -7 -> 5 -> -6 -> 6. In
         * this case, you should first remove 3 -> 4 -> -7, then -6 -> 6, leaving only
         * 5.
         */
        System.out.println("========= Q99 =========");
        ListNode headToRemoveConsecutiveSumZero = new ListNode(3);
        headToRemoveConsecutiveSumZero.next = new ListNode(4);
        headToRemoveConsecutiveSumZero.next.next = new ListNode(-7);
        headToRemoveConsecutiveSumZero.next.next.next = new ListNode(5);
        headToRemoveConsecutiveSumZero.next.next.next.next = new ListNode(-6);
        headToRemoveConsecutiveSumZero.next.next.next.next.next = new ListNode(6);

        System.out.print("Original Linked List: ");
        printLinkedList(headToRemoveConsecutiveSumZero);

        ListNode updatedHead = removeZeroSumSublists(headToRemoveConsecutiveSumZero);

        System.out.print("Updated Linked List: ");
        printLinkedList(updatedHead);

        /*
         * Q100.
         * Given a binary search tree, find the floor and ceiling of a given integer.
         * The floor is the highest element in the tree less than or equal to an
         * integer, while the ceiling is the lowest element in the tree greater than or
         * equal to an integer.
         * If either value does not exist, return None.
         */
        System.out.println("========= Q100 =========");
        TreeNode<Integer> treeToFindFloorOrCeiling = null;
        treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 8);
        insert(treeToFindFloorOrCeiling, 4);
        insert(treeToFindFloorOrCeiling, 12);
        insert(treeToFindFloorOrCeiling, 2);
        insert(treeToFindFloorOrCeiling, 6);
        insert(treeToFindFloorOrCeiling, 10);
        insert(treeToFindFloorOrCeiling, 14);

        int targetForFloorOrCeiling = 7;
        Integer floor = findFloor(treeToFindFloorOrCeiling, targetForFloorOrCeiling);
        Integer ceiling = findCeiling(treeToFindFloorOrCeiling, targetForFloorOrCeiling);

        System.out.println("Target: " + targetForFloorOrCeiling);
        System.out.println("Floor: " + (floor != null ? floor : "None"));
        System.out.println("Ceiling: " + (ceiling != null ? ceiling : "None"));

        /*
         * Q101.
         * Write an algorithm that finds the total number of set bits in all integers
         * between 1 and N.
         */
        System.out.println("========= Q101 =========");
        int nToGetNumOfBits = 10;
        int setBitsCount = countSetBits(nToGetNumOfBits);

        System.out.println("Total number of set bits between 1 and " + nToGetNumOfBits + ": " + setBitsCount);

        /*
         * Q102.
         * Given a array that's sorted but rotated at some unknown pivot, in which all
         * elements are distinct, find a "peak" element in O(log N) time.
         * An element is considered a peak if it is greater than both its left and right
         * neighbors. It is guaranteed that the first and last elements are lower than
         * all others.
         */
        System.out.println("========= Q102 =========");
        int[] numsToFindPeak = { 5, 6, 7, 8, 9, 10, 1, 2, 3, 4 };
        int peakElementIndex = findPeakElement(numsToFindPeak);

        System.out.println("Peak element: " + numsToFindPeak[peakElementIndex]);

        /*
         * Q103.
         * You are given a 2 x N board, and instructed to completely cover the board
         * with the following shapes:
         * Dominoes, or 2 x 1 rectangles.
         * Trominoes, or L-shapes.
         * For example, if N = 4, here is one possible configuration, where A is a
         * domino, and B and C are trominoes.
         * A B B C
         * A B C C
         * Given an integer N, determine in how many ways this task is possible.
         */
        System.out.println("========= Q103 =========");
        int colsForBoard = 4;
        int ways = countWaysToCoverBoard(colsForBoard);

        System.out.println("Number of ways to cover the board: " + ways);

        /*
         * Q104.
         * In linear algebra, a Toeplitz matrix is one in which the elements on any
         * given diagonal from top left to bottom right are identical.
         * Here is an example:
         * 1 2 3 4 8
         * 5 1 2 3 4
         * 4 5 1 2 3
         * 7 4 5 1 2
         * Write a program to determine whether a given input is a Toeplitz matrix.
         */
        System.out.println("========= Q104 =========");
        int[][] toeplitzMatrix = {
                { 1, 2, 3, 4, 8 },
                { 5, 1, 2, 3, 4 },
                { 4, 5, 1, 2, 3 },
                { 7, 4, 5, 1, 2 }
        };

        boolean isToeplitz = isToeplitzMatrix(toeplitzMatrix);

        if (isToeplitz) {
            System.out.println("The matrix is a Toeplitz matrix.");
        } else {
            System.out.println("The matrix is not a Toeplitz matrix.");
        }

        /*
         * Q105.
         * Consider the following scenario: there are N mice and N holes placed at
         * integer points along a line. Given this, find a method that maps mice to
         * holes such that the largest number of steps any mouse takes is minimized.
         * Each move consists of moving one mouse one unit to the left or right, and
         * only one mouse can fit inside each hole.
         * For example, suppose the mice are positioned at [1, 4, 9, 15], and the holes
         * are located at [10, -5, 0, 16]. In this case, the best pairing would require
         * us to send the mouse at 1 to the hole at -5, so our function should return 6.
         */
        System.out.println("========= Q105 =========");
        int[] mice = { 1, 4, 9, 15 };
        int[] holes = { 10, -5, 0, 16 };

        int minimumSteps = minimizeSteps(mice, holes);

        System.out.println("The minimum number of steps required is: " + minimumSteps);

        /*
         * Q106.
         * The United States uses the imperial system of weights and measures, which
         * means that there are many different, seemingly arbitrary units to measure
         * distance. There are 12 inches in a foot, 3 feet in a yard, 22 yards in a
         * chain, and so on.
         * Create a data structure that can efficiently convert a certain quantity of
         * one unit to the correct amount of any other unit. You should also allow for
         * additional units to be added to the system.
         */
        System.out.println("========= Q106 =========");
        UnitConverter unitConverter = new UnitConverter();
        unitConverter.addUnitConversion("inch", "foot", 0.0833);
        unitConverter.addUnitConversion("foot", "yard", 0.3333);
        unitConverter.addUnitConversion("yard", "mile", 0.00056818);

        double quantity = 1000;
        String fromUnit = "inch";
        String toUnit = "yard";
        double convertedResult = unitConverter.convert(quantity, fromUnit, toUnit);

        System.out.println(quantity + " " + fromUnit + " is equal to " + convertedResult + " " + toUnit);

        /*
         * Q107.
         * Write a program to merge two binary trees. Each node in the new tree should
         * hold a value equal to the sum of the values of the corresponding nodes of the
         * input trees.
         * If only one input tree has a node in a given position, the corresponding node
         * in the new tree should match that input node.
         */
        System.out.println("========= Q107 =========");
        TreeNode<Integer> t1 = new TreeNode<>(1);
        t1.left = new TreeNode<>(3);
        t1.right = new TreeNode<>(2);
        t1.left.left = new TreeNode<>(5);

        TreeNode<Integer> t2 = new TreeNode<>(2);
        t2.left = new TreeNode<>(1);
        t2.right = new TreeNode<>(3);
        t2.left.right = new TreeNode<>(4);
        t2.right.right = new TreeNode<>(7);

        TreeNode<Integer> mergedTree = mergeTrees(t1, t2);

        System.out.print("Merged tree (Preorder traversal): ");
        printPreorder(mergedTree);

        /*
         * Q108.
         * Given integers M and N, write a program that counts how many positive integer
         * pairs (a, b) satisfy the following conditions:
         * a + b = M
         * a XOR b = N
         */
        System.out.println("========= Q108 =========");
        int M = 40;
        int N = 20;

        int pairCount = countPairs(M, N);
        System.out.println("Number of positive integer pairs: " + pairCount);

        /*
         * Q109.
         * The 24 game is played as follows. You are given a list of four integers, each
         * between 1 and 9, in a fixed order. By placing the operators +, -, *, and /
         * between the numbers, and grouping them with parentheses, determine whether it
         * is possible to reach the value 24.
         * For example, given the input [5, 2, 7, 8], you should return True, since (5 *
         * 2 - 7) * 8 = 24.
         * Write a function that plays the 24 game.
         */
        System.out.println("========= Q109 =========");
        int[] numsFor24Game = { 5, 2, 7, 8 };
        boolean canReach24 = canReach24(numsFor24Game);
        System.out.println("Can reach 24: " + canReach24);

        /*
         * Q110.
         * Given an array of numbers and a number k, determine if there are three
         * entries in the array which add up to the specified number k. For example,
         * given [20, 303, 3, 4, 25] and k = 49, return true as 20 + 4 + 25 = 49.
         */
        System.out.println("========= Q110 =========");
        int[] numsForThreeSum = { 20, 303, 3, 4, 25 };
        int kForThreeSum = 49;
        boolean hasThreeSum = hasThreeSum(numsForThreeSum, kForThreeSum);
        System.out.println("Has three sum: " + hasThreeSum);

        /*
         * Q111.
         * Given a set of points (x, y) on a 2D cartesian plane, find the two closest
         * points. For example, given the points [(1, 1), (-1, -1), (3, 4), (6, 1), (-1,
         * -6), (-4, -3)], return [(-1, -1), (1, 1)].
         */
        System.out.println("========= Q111 =========");
        Point[] setOfPoints = {
                new Point(1, 1),
                new Point(-1, -1),
                new Point(3, 4),
                new Point(6, 1),
                new Point(-1, -6),
                new Point(-4, -3)
        };

        Point[] closestPoints = findClosestPoints(setOfPoints);
        System.out.println(Arrays.toString(closestPoints));

        /*
         * Q112.
         * You are given an N by N matrix of random letters and a dictionary of words.
         * Find the maximum number of words that can be packed on the board from the
         * given dictionary.
         * A word is considered to be able to be packed on the board if:
         * It can be found in the dictionary
         * It can be constructed from untaken letters by other words found so far on the
         * board
         * The letters are adjacent to each other (vertically and horizontally, not
         * diagonally).
         * Each tile can be visited only once by any word.
         * For example, given the following dictionary:
         * { 'eat', 'rain', 'in', 'rat' }
         * and matrix:
         * [['e', 'a', 'n'],
         * ['t', 't', 'i'],
         * ['a', 'r', 'a']]
         * Your function should return 3, since we can make the words 'eat', 'in', and
         * 'rat' without them touching each other. We could have alternatively made
         * 'eat' and 'rain', but that would be incorrect since that's only 2 words.
         */
        System.out.println("========= Q112 =========");
        char[][] randomLetterBoard = {
                { 'e', 'a', 'n' },
                { 't', 't', 'i' },
                { 'a', 'r', 'a' }
        };

        String[] dictionaryOfWords = { "eat", "rain", "in", "rat" };

        int maxPackedWords = findMaxPackedWords(randomLetterBoard, dictionaryOfWords);
        System.out.println("Maximum number of words that can be packed: " + maxPackedWords);

        /*
         * Q113.
         * You are given a string of length N and a parameter k. The string can be
         * manipulated by taking one of the first k letters and moving it to the end.
         * Write a program to determine the lexicographically smallest string that can
         * be created after an unlimited number of moves.
         * For example, suppose we are given the string daily and k = 1. The best we can
         * create in this case is ailyd.
         */
        System.out.println("========= Q113 =========");
        String inputToFindSmallestString = "daily";
        int firstNumOfLetters = 1;

        String smallestString = getLexicographicallySmallestString(inputToFindSmallestString, firstNumOfLetters);
        System.out.println("Lexicographically Smallest String: " + smallestString);

        /*
         * Q114.
         * A ternary search tree is a trie-like data structure where each node may have
         * up to three children. Here is an example which represents the words code,
         * cob, be, ax, war, and we.
         * "       c           "
         * "    /  |  \        "
         * "   b   o   w       "
         * " / |   |   |       "
         * "a  e   d   a       "
         * "|    / |   | \     "
         * "x   b  e   r  e    "
         * The tree is structured according to the following rules:
         * left child nodes link to words lexicographically earlier than the parent
         * prefix
         * right child nodes link to words lexicographically later than the parent
         * prefix
         * middle child nodes continue the current word
         * For instance, since code is the first word inserted in the tree, and cob
         * lexicographically precedes cod, cob is represented as a left child extending
         * from cod.
         * Implement insertion and search functions for a ternary search tree.
         */
        System.out.println("========= Q114 =========");
        TernarySearchTree ternarySearchTree = new TernarySearchTree();

        ternarySearchTree.insert("code");
        ternarySearchTree.insert("cob");
        ternarySearchTree.insert("be");
        ternarySearchTree.insert("ax");
        ternarySearchTree.insert("war");
        ternarySearchTree.insert("we");

        System.out.println(ternarySearchTree.search("code")); // Output: true
        System.out.println(ternarySearchTree.search("cob")); // Output: true
        System.out.println(ternarySearchTree.search("be")); // Output: true
        System.out.println(ternarySearchTree.search("ax")); // Output: true
        System.out.println(ternarySearchTree.search("war")); // Output: true
        System.out.println(ternarySearchTree.search("we")); // Output: true
        System.out.println(ternarySearchTree.search("hello")); // Output: false

        /*
         * Q115.
         * A typical American-style crossword puzzle grid is an N x N matrix with black
         * and white squares, which obeys the following rules:
         * Every white square must be part of an "across" word and a "down" word.
         * No word can be fewer than three letters long.
         * Every white square must be reachable from every other white square.
         * The grid is rotationally symmetric (for example, the colors of the top left
         * and bottom right squares must match).
         * Write a program to determine whether a given matrix qualifies as a crossword
         * grid.
         */
        System.out.println("========= Q115 =========");
        char[][] grid1 = {
                { 'B', 'B', 'B', 'B', 'B' },
                { 'B', 'W', 'W', 'W', 'B' },
                { 'B', 'W', 'O', 'W', 'B' },
                { 'B', 'W', 'W', 'W', 'B' },
                { 'B', 'B', 'B', 'B', 'B' }
        };

        char[][] grid2 = {
                { 'B', 'B', 'B', 'B', 'B' },
                { 'B', 'W', 'B', 'W', 'B' },
                { 'B', 'W', 'W', 'W', 'B' },
                { 'B', 'B', 'B', 'W', 'B' },
                { 'B', 'B', 'B', 'W', 'B' }
        };
        char[][] grid3 = {
                { 'B', 'B', 'B', 'B', 'B' },
                { 'B', 'W', 'B', 'W', 'B' },
                { 'B', 'W', 'B', 'W', 'B' },
                { 'B', 'W', 'B', 'W', 'B' },
                { 'B', 'B', 'B', 'W', 'B' }
        };

        System.out.println("Given grid is a crossword grid: " + isCrosswordGrid(grid1));
        System.out.println("Given grid is a crossword grid: " + isCrosswordGrid(grid2));
        System.out.println("Given grid is a crossword grid: " + isCrosswordGrid(grid3));
        /*
         * Q116.
         * You are given a string formed by concatenating several words corresponding to
         * the integers zero through nine and then anagramming.
         * For example, the input could be 'niesevehrtfeev', which is an anagram of
         * 'threefiveseven'. Note that there can be multiple instances of each integer.
         * Given this string, return the original integers in sorted order. In the
         * example above, this would be 357.
         */
        System.out.println("========= Q116 =========");
        String s = "niesevehrtfeev";
        String originalDigits = getOriginalDigits(s);
        System.out.println(originalDigits); // Output: 357

        /*
         * Q117.
         * A strobogrammatic number is a positive number that appears the same after
         * being rotated 180 degrees. For example, 16891 is strobogrammatic.
         * Create a program that finds all strobogrammatic numbers with N digits.
         */
        System.out.println("========= Q117 =========");
        int nDigitToFindStrobogrammatic = 5;
        List<String> strobogrammaticNumbers = findStrobogrammaticNumbers(nDigitToFindStrobogrammatic);
        System.out.println(
                "Strobogrammatic numbers with " + nDigitToFindStrobogrammatic + " digits: " + strobogrammaticNumbers);

        /*
         * Q118.
         * The “active time” of a courier is the time between the pickup and dropoff of
         * a delivery. Given a set of data formatted like the following:
         * (delivery id, timestamp, pickup/dropoff)
         * Calculate the total active time in seconds. A courier can pick up multiple
         * orders before dropping them off. The timestamp is in unix epoch seconds.
         * For example, if the input is the following:
         * (1, 1573280047, 'pickup')
         * (1, 1570320725, 'dropoff')
         * (2, 1570321092, 'pickup')
         * (3, 1570321212, 'pickup')
         * (3, 1570322352, 'dropoff')
         * (2, 1570323012, 'dropoff')
         * The total active time would be 1260 seconds.
         */
        System.out.println("========= Q118 =========");
        String[] courierSchedule = {
                "(1, 1573280047, 'pickup')",
                "(1, 1570320725, 'dropoff')",
                "(2, 1570321092, 'pickup')",
                "(3, 1570321212, 'pickup')",
                "(3, 1570322352, 'dropoff')",
                "(2, 1570323012, 'dropoff')"
        };

        int totalActiveTime = calculateTotalActiveTime(courierSchedule);
        System.out.println("Total active time: " + totalActiveTime + " seconds");

        /*
         * Q119.
         * Write a function that takes a natural number as input and returns the number
         * of digits the input has.
         * Constraint: don't use any loops.
         */
        System.out.println("========= Q119 =========");
        int number = 12345;
        int digitCount = countDigits(number);
        System.out.println("Number of digits: " + digitCount);

        /*
         * Q120.
         * You are writing an AI for a 2D map game. You are somewhere in a 2D grid, and
         * there are coins strewn about over the map.
         * Given the position of all the coins and your current position, find the
         * closest coin to you in terms of Manhattan distance. That is, you can move
         * around up, down, left, and right, but not diagonally. If there are multiple
         * possible closest coins, return any of them.
         * For example, given the following map, where you are x, coins are o, and empty
         * spaces are . (top left is 0, 0):
         * ---------------------
         * | . | . | x | . | o |
         * ---------------------
         * | o | . | . | . | . |
         * ---------------------
         * | o | . | . | . | o |
         * ---------------------
         * | . | . | o | . | . |
         * ---------------------
         * return (0, 4), since that coin is closest. This map would be represented in
         * our question as:
         * Our position: (0, 2)
         * Coins: [(0, 4), (1, 0), (2, 0), (3, 2)]
         */
        System.out.println("========= Q120 =========");
        char[][] coinMap = {
                { '.', '.', 'x', '.', 'o' },
                { 'o', '.', '.', '.', '.' },
                { 'o', '.', '.', '.', 'o' },
                { '.', '.', 'o', '.', '.' }
        };

        int[] currentPosition = { 0, 2 };

        int[] closestCoin = findClosestCoin(currentPosition, coinMap);
        System.out.println("Closest coin: (" + closestCoin[0] + ", " + closestCoin[1] + ")");

        /*
         * Q121.
         * Given a string, generate all possible subsequences of the string.
         * For example, given the string xyz, return an array or set with the following
         * strings:
         * x
         * y
         * z
         * xy
         * xz
         * yz
         * xyz
         * Note that zx is not a valid subsequence since it is not in the order of the
         * given string.
         */
        System.out.println("========= Q121 =========");
        String str = "xyz";
        List<String> subsequences = generateSubsequences(str);
        System.out.println(subsequences);

        /*
         * Q122.
         * Read this Wikipedia article on Base64 encoding.
         * Implement a function that converts a hex string to base64.
         * For example, the string:
         * deadbeef
         * should produce:
         * 3q2+7w==
         */
        System.out.println("========= Q122 =========");
        String hexStringToConvert = "deadbeef";
        String base64String = convertHexToBase64(hexStringToConvert);
        System.out.println("Base64 encoded string: " + base64String);

        /*
         * Q123.
         * Yesterday you implemented a function that encodes a hexadecimal string into
         * Base64.
         * Write a function to decode a Base64 string back to a hexadecimal string.
         * For example, the following string:
         * 3q2+7w==
         * should produce:
         * deadbeef
         */
        System.out.println("========= Q123 =========");
        String base64StringToConvert = "3q2+7w==";
        String hexString = convertBase64ToHex(base64StringToConvert);
        System.out.println("Hexadecimal string: " + hexString);

        /*
         * Q124.
         * Given a string, sort it in decreasing order based on the frequency of
         * characters. If there are multiple possible solutions, return any of them.
         * For example, given the string tweet, return tteew. eettw would also be
         * acceptable.
         */
        System.out.println("========= Q124 =========");
        String stringToDecreasingOrder = "tweet";
        String sortedStr = sortStringByFrequency(stringToDecreasingOrder);
        System.out.println(sortedStr);

        /*
         * Q125.
         * Given a binary tree and an integer k, return whether there exists a
         * root-to-leaf path that sums up to k.
         * For example, given k = 18 and the following binary tree:
         * "    8      "
         * "   / \     "
         * "  4   13   "
         * " / \   \   "
         * "2   6   19 "
         * Return True since the path 8 -> 4 -> 6 sums to 18.
         */
        System.out.println("========= Q125 =========");
        TreeNode<Integer> treeToFindTargetSum = new TreeNode<>(8);
        treeToFindTargetSum.left = new TreeNode<>(4);
        treeToFindTargetSum.right = new TreeNode<>(13);
        treeToFindTargetSum.left.left = new TreeNode<>(2);
        treeToFindTargetSum.left.right = new TreeNode<>(6);
        treeToFindTargetSum.right.right = new TreeNode<>(19);

        int targetSum = 18;
        boolean hasPathSum = hasPathSum(treeToFindTargetSum, targetSum);
        System.out.println(hasPathSum); // Output: true

        /*
         * Q126.
         * Using a function rand5() that returns an integer from 1 to 5 (inclusive) with
         * uniform probability, implement a function rand7() that returns an integer
         * from 1 to 7 (inclusive).
         */
        System.out.println("========= Q126 =========");
        for (int i = 0; i < 10; i++) {
            System.out.println(rand7());
        }

        /*
         * Q127.
         * Given a positive integer N, find the smallest number of steps it will take to
         * reach 1.
         * There are two kinds of permitted steps:
         * You may decrement N to N - 1.
         * If a * b = N, you may decrement N to the larger of a and b.
         * For example, given 100, you can reach 1 in five steps with the following
         * route: 100 -> 10 -> 9 -> 3 -> 2 -> 1.
         */
        System.out.println("========= Q127 =========");
        int startToFindMinStepsTo1 = 100;
        int minStepsTo1 = minStepsToOne(startToFindMinStepsTo1);
        System.out.println("Minimum number of steps to reach 1: " + minStepsTo1);
    }

    private static void printLinkedList(ListNode head) {
        ListNode current = head;
        while (current != null) {
            System.out.print(current.val + " -> ");
            current = current.next;
        }
        System.out.println("null");
    }

    private static <T> void printTree(TreeNode<T> node) {
        if (node == null) {
            return;
        }

        System.out.print(node.val + " ");
        printTree(node.left);
        printTree(node.right);
    }

    private static <T> void printPreorder(TreeNode<T> root) {
        if (root != null) {
            System.out.print(root.val + " ");
            printPreorder(root.left);
            printPreorder(root.right);
        }
    }

    // S55.
    public static boolean checkOverlap(Rectangle[] rectangles) {
        int n = rectangles.length;

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (rectangles[i].overlaps(rectangles[j])) {
                    return true;
                }
            }
        }

        return false;
    }
}