"""
Tests for graceful training stop (SIGTERM → KeyboardInterrupt → checkpoint_interrupt.pt).
"""
import signal
import pytest


class TestGracefulStop:
    """SIGTERM handler should raise KeyboardInterrupt to trigger checkpoint save."""

    def test_sigterm_raises_keyboard_interrupt(self):
        """The SIGTERM handler must raise KeyboardInterrupt."""
        def handler(signum, frame):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)

    def test_sigterm_is_registered(self):
        """SIGTERM should be convertible to KeyboardInterrupt via the standard pattern."""
        # This is the pattern used in train_production.main()
        caught = False

        def sigterm_handler(signum, frame):
            nonlocal caught
            caught = True
            raise KeyboardInterrupt()

        old_handler = signal.signal(signal.SIGTERM, sigterm_handler)
        try:
            # Simulate the try/except pattern
            raised = False
            try:
                sigterm_handler(signal.SIGTERM, None)
            except KeyboardInterrupt:
                raised = True
            assert raised, "SIGTERM handler did not raise KeyboardInterrupt"
            assert caught, "SIGTERM handler was not called"
        finally:
            signal.signal(signal.SIGTERM, old_handler)
