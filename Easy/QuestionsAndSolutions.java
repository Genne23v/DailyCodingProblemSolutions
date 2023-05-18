import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
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

    public static void main(String[] args) throws InterruptedException {
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

        /*
         * Q31.
         * Determine whether a doubly linked list is a palindrome. What if it’s singly
         * linked?
         * For example, 1 -> 4 -> 3 -> 4 -> 1 returns True while 1 -> 4 returns False.
         */
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
        TreeNode<Integer> findTwoSumBinaryTree = new TreeNode<>(10);
        findTwoSumBinaryTree.left = new TreeNode<>(5);
        findTwoSumBinaryTree.right = new TreeNode<>(15);
        findTwoSumBinaryTree.right.left = new TreeNode<>(11);
        findTwoSumBinaryTree.right.right = new TreeNode<>(15);

        TreeNode<Integer>[] twoNodesForK = findTarget(binaryToPrint, 20);
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
        int[] arrayToRotate = { 1, 2, 3, 4, 5 };
        int numOfRotation = 3;

        rotate(arrayToRotate, numOfRotation);

        System.out.println("Rotated array:");
        for (int numToRotate : arrayToRotate) {
            System.out.print(numToRotate + " ");
        }

    }

    private static void printLinkedList(ListNode head) {
        ListNode current = head;
        while (current != null) {
            System.out.print(current.val + " -> ");
            current = current.next;
        }
        System.out.println("null");
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
