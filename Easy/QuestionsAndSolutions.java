import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.LinkedList;
import java.util.Map;
import java.util.Stack;
import java.util.Collections;
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
                start = (start + 1) % buffer.length; // When the buffer is full, use start to circulate the buffer array
            }
        }

        public int get_last(int i) {
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
                Set<Set<Integer>> newSubsets = new HashSet<>(); // Each iteration returns new Sets with element
                for (Set<Integer> subset : powerSet) {
                    Set<Integer> newSubset = new HashSet<>(subset);
                    newSubset.add(element); // Additional Set with new element
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

            // Otherwise, move on to the next number
            num += 9;
        }

        // This line is unreachable, but required by Java
        return -1;
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

    }
}
