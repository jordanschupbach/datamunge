package octra

import "testing"

func TestHello(t *testing.T) {
	Hello()
}

func TestDPair(t *testing.T) {
	dp := NewDPair(1.0, 2.0)
	if dp.GetFirst() != 1.0 {
		t.Errorf("Expected GetFirst() = 1, got %v", dp.GetFirst())
	}
	if dp.GetSecond() != 2.0 {
		t.Errorf("Expected GetSecond() = 2, got %v", dp.GetSecond())
	}
	DeleteDPair(dp)
}

func TestSVector(t *testing.T) {
	s := NewSVector(int64(3))
	s.Set(0, "1")
	s.Set(1, "2")
	s.Set(2, "3")
	if s.Get(0) != "1" {
		t.Errorf("Expected Get(0) = '1', got %v", s.Get(0))
	}
	if s.Get(1) != "2" {
		t.Errorf("Expected Get(1) = '2', got %v", s.Get(1))
	}
	if s.Get(2) != "3" {
		t.Errorf("Expected Get(2) = '3', got %v", s.Get(2))
	}
	DeleteSVector(s)
}
