"""Comprehensive test suite for Quantro Heart Model simulation framework.

Tests all models, integrators, parameter sweeps, security features, and edge cases.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from quantro_simulator import (
    SimulationConfig,
    ModelType,
    OverlayMode,
    create_model,
    RK4Integrator,
    run_parameter_sweep,
    run_comprehensive_simulation,
    MichaelisMentenModel,
    SIRModel,
    FitzHughNagumoModel,
    NernstModel,
    PoiseuilleFlowModel,
)

from security_config import (
    AccessLevel,
    AuditEventType,
    AuditLogger,
    AccessControl,
    DataAnonymizer,
    SecureSimulationSession,
)

try:
    from security_config import DataEncryption, CRYPTO_AVAILABLE
except ImportError:
    CRYPTO_AVAILABLE = False


class TestSimulationModels(unittest.TestCase):
    """Test all cardiac models."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(
            model=ModelType.SIR,
            overlay_mode=OverlayMode.BASELINE,
            t_start=0.0,
            t_end=10.0,
            dt=0.01
        )

    def test_michaelis_menten_model(self):
        """Test Michaelis-Menten model initialization and integration."""
        config = SimulationConfig(
            model=ModelType.MICHAELIS_MENTEN,
            overlay_mode=OverlayMode.BASELINE
        )
        model = create_model(config)
        self.assertIsInstance(model, MichaelisMentenModel)

        # Test integration
        time_points, trajectory = RK4Integrator.integrate(model, lambda_param=0.0)
        self.assertEqual(len(time_points), config.num_steps + 1)
        self.assertEqual(trajectory.shape[0], config.num_steps + 1)

        # Substrate should decrease over time
        self.assertGreater(trajectory[0, 0], trajectory[-1, 0])

    def test_sir_model(self):
        """Test SIR compartmental model."""
        config = SimulationConfig(
            model=ModelType.SIR,
            overlay_mode=OverlayMode.BASELINE
        )
        model = create_model(config)
        self.assertIsInstance(model, SIRModel)

        time_points, trajectory = RK4Integrator.integrate(model, lambda_param=0.0)

        # Test conservation: S + I + R should be approximately 1
        totals = np.sum(trajectory, axis=1)
        np.testing.assert_allclose(totals, 1.0, rtol=1e-3)

        # Susceptible should decrease
        self.assertGreater(trajectory[0, 0], trajectory[-1, 0])

    def test_fitzhugh_nagumo_model(self):
        """Test FitzHugh-Nagumo excitable model."""
        config = SimulationConfig(
            model=ModelType.FITZHUGH_NAGUMO,
            overlay_mode=OverlayMode.BASELINE,
            t_end=50.0
        )
        model = create_model(config)
        self.assertIsInstance(model, FitzHughNagumoModel)

        time_points, trajectory = RK4Integrator.integrate(model, lambda_param=0.3)

        # Check that we get oscillations (action potentials)
        v = trajectory[:, 0]
        self.assertTrue(np.max(v) > 0.5)  # Should have excitation
        self.assertTrue(np.min(v) < -0.5)  # Should have rest

    def test_nernst_model(self):
        """Test Nernst potential model."""
        config = SimulationConfig(
            model=ModelType.NERNST,
            overlay_mode=OverlayMode.BASELINE
        )
        model = create_model(config)
        self.assertIsInstance(model, NernstModel)

        time_points, trajectory = RK4Integrator.integrate(model, lambda_param=0.0)

        # Check concentrations remain positive
        self.assertTrue(np.all(trajectory[:, 0] > 0))
        self.assertTrue(np.all(trajectory[:, 1] > 0))

        # Test potential calculation
        potential = model.calculate_potential(trajectory[-1])
        self.assertIsInstance(potential, float)

    def test_poiseuille_model(self):
        """Test Poiseuille flow model."""
        config = SimulationConfig(
            model=ModelType.POISEUILLE,
            overlay_mode=OverlayMode.BASELINE
        )
        model = create_model(config)
        self.assertIsInstance(model, PoiseuilleFlowModel)

        time_points, trajectory = RK4Integrator.integrate(model, lambda_param=0.0)

        # Pressure should be positive
        self.assertTrue(np.all(trajectory[:, 0] > 0))


class TestOverlayModes(unittest.TestCase):
    """Test overlay mode functionality."""

    def test_all_overlay_modes(self):
        """Test that all overlay modes produce different results."""
        modes = list(OverlayMode)
        results = {}

        for mode in modes:
            config = SimulationConfig(
                model=ModelType.SIR,
                overlay_mode=mode,
                t_end=10.0
            )
            model = create_model(config)
            _, trajectory = RK4Integrator.integrate(model, lambda_param=0.5)
            results[mode] = trajectory[-1, 0]

        # Different modes should produce different results
        baseline = results[OverlayMode.BASELINE]
        self.assertNotEqual(results[OverlayMode.RESIDUAL], baseline)
        self.assertNotEqual(results[OverlayMode.PARAM_MOD], baseline)
        self.assertNotEqual(results[OverlayMode.CONTROL], baseline)


class TestParameterSweeps(unittest.TestCase):
    """Test parameter sweep functionality."""

    def test_basic_parameter_sweep(self):
        """Test single parameter sweep."""
        lambda_values = np.linspace(0.0, 1.0, 10)
        config = SimulationConfig(
            model=ModelType.MICHAELIS_MENTEN,
            overlay_mode=OverlayMode.BASELINE,
            lambda_values=lambda_values
        )

        results = run_parameter_sweep(config)

        self.assertEqual(len(results), len(lambda_values))

        # Check that results vary with lambda
        val1_array = np.array([r.val1 for r in results])
        self.assertGreater(np.std(val1_array), 0)

    def test_comprehensive_simulation(self):
        """Test comprehensive simulation across all models and modes."""
        # Use smaller parameter space for faster testing
        results_df = run_comprehensive_simulation()

        # Should have results for all combinations
        n_models = len(ModelType)
        n_modes = len(OverlayMode)
        n_lambda = 12  # default
        expected_rows = n_models * n_modes * n_lambda

        self.assertEqual(len(results_df), expected_rows)

        # Check required columns
        required_cols = ['model', 'mode', 'lambda', 'val1', 'val2', 'val3']
        for col in required_cols:
            self.assertIn(col, results_df.columns)


class TestIntegrator(unittest.TestCase):
    """Test RK4 integrator accuracy and stability."""

    def test_rk4_conservation(self):
        """Test that RK4 conserves quantities in SIR model."""
        config = SimulationConfig(
            model=ModelType.SIR,
            overlay_mode=OverlayMode.BASELINE,
            dt=0.001  # Fine time step
        )
        model = create_model(config)
        time_points, trajectory = RK4Integrator.integrate(model, lambda_param=0.0)

        # Check conservation of total population
        totals = np.sum(trajectory, axis=1)
        np.testing.assert_allclose(totals, 1.0, rtol=1e-3)

    def test_rk4_stability(self):
        """Test integrator stability with different time steps."""
        dt_values = [0.1, 0.01, 0.001]
        results = []

        for dt in dt_values:
            config = SimulationConfig(
                model=ModelType.SIR,
                overlay_mode=OverlayMode.BASELINE,
                dt=dt,
                t_end=10.0
            )
            model = create_model(config)
            _, trajectory = RK4Integrator.integrate(model, lambda_param=0.0)
            results.append(trajectory[-1, 0])

        # Results should converge as dt decreases
        self.assertGreater(abs(results[0] - results[1]), abs(results[1] - results[2]))


class TestSecurityFeatures(unittest.TestCase):
    """Test HIPAA compliance and security features."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_audit_logging(self):
        """Test audit logger functionality."""
        log_path = self.temp_dir / "test_audit.log"
        audit_logger = AuditLogger(log_path)

        from security_config import AuditEvent

        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="test_user",
            resource="test_resource",
            action="read",
            success=True
        )

        audit_logger.log_event(event)

        # Check that log file was created
        self.assertTrue(log_path.exists())

        # Read and verify
        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("test_user", content)
            self.assertIn("test_resource", content)

    def test_access_control(self):
        """Test role-based access control."""
        ac = AccessControl()

        # Researcher should be able to run simulations
        self.assertTrue(ac.check_permission(
            AccessLevel.RESEARCHER,
            'run_simulation'
        ))

        # Researcher should NOT be able to read PHI
        self.assertFalse(ac.check_permission(
            AccessLevel.RESEARCHER,
            'read_phi'
        ))

        # Clinician should be able to read PHI
        self.assertTrue(ac.check_permission(
            AccessLevel.CLINICIAN,
            'read_phi'
        ))

        # Admin should have all permissions
        self.assertTrue(ac.check_permission(
            AccessLevel.ADMIN,
            'manage_users'
        ))

    def test_data_anonymization(self):
        """Test data anonymization functions."""
        # Test patient ID anonymization
        patient_id = "PATIENT-12345"
        anon_id = DataAnonymizer.anonymize_patient_id(patient_id)

        self.assertIsInstance(anon_id, str)
        self.assertNotEqual(anon_id, patient_id)
        self.assertEqual(len(anon_id), 16)

        # Same input should give same output
        anon_id2 = DataAnonymizer.anonymize_patient_id(patient_id)
        self.assertEqual(anon_id, anon_id2)

        # Different salt should give different output
        anon_id3 = DataAnonymizer.anonymize_patient_id(patient_id, salt="different")
        self.assertNotEqual(anon_id, anon_id3)

        # Test PHI removal
        data_with_phi = {
            'patient_name': 'John Doe',
            'patient_id': '12345',
            'age': 45,
            'diagnosis': 'Arrhythmia'
        }

        anonymized = DataAnonymizer.remove_phi_fields(data_with_phi)
        self.assertNotIn('patient_name', anonymized)
        self.assertNotIn('patient_id', anonymized)
        self.assertIn('age', anonymized)
        self.assertIn('diagnosis', anonymized)

    @unittest.skipIf(not CRYPTO_AVAILABLE, "Cryptography package not installed")
    def test_encryption(self):
        """Test data encryption and decryption."""
        key_path = self.temp_dir / "test_key"
        encryptor = DataEncryption(key_path)

        # Test basic encryption
        plaintext = b"Sensitive patient data"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        self.assertNotEqual(encrypted, plaintext)
        self.assertEqual(decrypted, plaintext)

        # Test file encryption
        input_file = self.temp_dir / "test_input.txt"
        encrypted_file = self.temp_dir / "test_encrypted.bin"
        decrypted_file = self.temp_dir / "test_decrypted.txt"

        input_file.write_text("Test data for encryption")

        encryptor.encrypt_file(input_file, encrypted_file)
        encryptor.decrypt_file(encrypted_file, decrypted_file)

        self.assertEqual(
            input_file.read_text(),
            decrypted_file.read_text()
        )

    def test_secure_session(self):
        """Test secure session management."""
        session = SecureSimulationSession(
            user_id="test_user",
            user_level=AccessLevel.RESEARCHER,
            session_timeout=3600
        )

        # Session should be valid
        self.assertTrue(session.is_valid())

        # Should be able to run simulation
        self.assertTrue(session.authorize_action("run_simulation", "SIR_model"))

        # Should NOT be able to read PHI
        self.assertFalse(session.authorize_action("read_phi", "patient_data"))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_zero_time_step(self):
        """Test behavior with very small time steps."""
        config = SimulationConfig(
            model=ModelType.SIR,
            overlay_mode=OverlayMode.BASELINE,
            dt=1e-6,
            t_end=0.001
        )
        model = create_model(config)
        time_points, trajectory = RK4Integrator.integrate(model, lambda_param=0.0)

        # Should complete without error
        self.assertGreater(len(time_points), 0)

    def test_extreme_lambda_values(self):
        """Test with extreme lambda values."""
        extreme_lambdas = [-10.0, 0.0, 10.0, 100.0]

        for lam in extreme_lambdas:
            config = SimulationConfig(
                model=ModelType.MICHAELIS_MENTEN,
                overlay_mode=OverlayMode.RESIDUAL,
                t_end=5.0
            )
            model = create_model(config)
            time_points, trajectory = RK4Integrator.integrate(model, lambda_param=lam)

            # Should complete without NaN or Inf
            self.assertFalse(np.any(np.isnan(trajectory)))
            self.assertFalse(np.any(np.isinf(trajectory)))

    def test_invalid_model_type(self):
        """Test handling of invalid model type."""
        config = SimulationConfig(
            model=ModelType.SIR,
            overlay_mode=OverlayMode.BASELINE
        )
        # Manually set to invalid type
        config.model = "INVALID_MODEL"

        with self.assertRaises(ValueError):
            create_model(config)


def run_test_suite():
    """Run the complete test suite with reporting."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSimulationModels))
    suite.addTests(loader.loadTestsFromTestCase(TestOverlayModes))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterSweeps))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrator))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    exit(0 if success else 1)
