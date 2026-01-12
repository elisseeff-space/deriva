"""Tests for C# language extractor."""

from __future__ import annotations

from deriva.adapters.treesitter import TreeSitterManager


class TestCSharpTypes:
    """Tests for C# type extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_class(self):
        """Should extract C# class."""
        source = """
public class UserService
{
    private string _name;
}
"""
        types = self.manager.extract_types(source, language="csharp")

        assert len(types) >= 1
        service = next((t for t in types if t.name == "UserService"), None)
        assert service is not None
        assert service.kind == "class"

    def test_extracts_class_with_inheritance(self):
        """Should extract class with base class."""
        source = """
public class Admin : User
{
    public int Level { get; set; }
}
"""
        types = self.manager.extract_types(source, language="csharp")

        admin = next((t for t in types if t.name == "Admin"), None)
        assert admin is not None

    def test_extracts_class_with_interface(self):
        """Should extract class implementing interface."""
        source = """
public class UserRepository : IRepository<User>, IDisposable
{
    public void Dispose() { }
}
"""
        types = self.manager.extract_types(source, language="csharp")

        repo = next((t for t in types if t.name == "UserRepository"), None)
        assert repo is not None

    def test_extracts_interface(self):
        """Should extract C# interface."""
        source = """
public interface IRepository<T>
{
    T GetById(int id);
    void Save(T entity);
}
"""
        types = self.manager.extract_types(source, language="csharp")

        repo = next((t for t in types if t.name == "IRepository"), None)
        assert repo is not None
        assert repo.kind == "interface"

    def test_extracts_struct(self):
        """Should extract C# struct."""
        source = """
public struct Point
{
    public int X;
    public int Y;
}
"""
        types = self.manager.extract_types(source, language="csharp")

        point = next((t for t in types if t.name == "Point"), None)
        assert point is not None

    def test_extracts_enum(self):
        """Should extract C# enum."""
        source = """
public enum Status
{
    Pending,
    Active,
    Completed
}
"""
        types = self.manager.extract_types(source, language="csharp")

        status = next((t for t in types if t.name == "Status"), None)
        assert status is not None

    def test_extracts_abstract_class(self):
        """Should extract abstract class."""
        source = """
public abstract class BaseEntity
{
    protected int Id { get; set; }
    public abstract void Validate();
}
"""
        types = self.manager.extract_types(source, language="csharp")

        entity = next((t for t in types if t.name == "BaseEntity"), None)
        assert entity is not None

    def test_extracts_record(self):
        """Should extract C# record."""
        source = """
public record User(string Name, int Age);
"""
        types = self.manager.extract_types(source, language="csharp")
        user = next((t for t in types if t.name == "User"), None)

        assert user is not None


class TestCSharpMethods:
    """Tests for C# method extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_public_method(self):
        """Should extract public method."""
        source = """
public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")

        add = next((m for m in methods if m.name == "Add"), None)
        assert add is not None

    def test_extracts_private_method(self):
        """Should extract private method."""
        source = """
public class Service
{
    private void Helper()
    {
        // internal logic
    }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")

        helper = next((m for m in methods if m.name == "Helper"), None)
        assert helper is not None

    def test_extracts_static_method(self):
        """Should extract static method."""
        source = """
public class Utils
{
    public static string Format(string input)
    {
        return input.Trim();
    }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")

        format_method = next((m for m in methods if m.name == "Format"), None)
        assert format_method is not None

    def test_extracts_async_method(self):
        """Should extract async method."""
        source = """
public class ApiClient
{
    public async Task<User> GetUserAsync(int id)
    {
        return await _repository.FindAsync(id);
    }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")

        get_user = next((m for m in methods if m.name == "GetUserAsync"), None)
        assert get_user is not None

    def test_extracts_generic_method(self):
        """Should handle methods with generics."""
        source = """
public class Repository
{
    public T FindById<T>(int id) where T : class
    {
        return null;
    }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")

        find_by_id = next((m for m in methods if m.name == "FindById"), None)
        assert find_by_id is not None

    def test_extracts_constructor(self):
        """Should extract constructor."""
        source = """
public class Person
{
    public Person(string name)
    {
        Name = name;
    }

    public string Name { get; }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")
        constructor = next((m for m in methods if m.name == "Person"), None)

        assert constructor is not None
        assert constructor.class_name == "Person"

    def test_extracts_class_with_property_accessors(self):
        """Should handle class with property accessors without crashing."""
        source = """
public class User
{
    private string _name;

    public string Name
    {
        get { return _name; }
        set { _name = value; }
    }

    public void SetName(string name) { _name = name; }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")
        set_name = next((m for m in methods if m.name == "SetName"), None)

        # Regular method should be extracted even with properties in class
        assert set_name is not None


class TestCSharpImports:
    """Tests for C# using statement extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_using(self):
        """Should extract using statement."""
        source = """
using System;
"""
        imports = self.manager.extract_imports(source, language="csharp")

        assert len(imports) >= 1

    def test_extracts_multiple_usings(self):
        """Should extract multiple using statements."""
        source = """
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
"""
        imports = self.manager.extract_imports(source, language="csharp")

        assert len(imports) >= 1

    def test_extracts_using_with_alias(self):
        """Should extract using with alias."""
        source = """
using Console = System.Console;
using Dict = System.Collections.Generic.Dictionary<string, object>;
"""
        imports = self.manager.extract_imports(source, language="csharp")

        assert len(imports) >= 1

    def test_extracts_global_using(self):
        """Should handle global using."""
        source = """
global using System;
global using System.Collections.Generic;
"""
        imports = self.manager.extract_imports(source, language="csharp")

        assert len(imports) >= 2


class TestCSharpEdgeCases:
    """Tests for edge cases in C# extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_handles_empty_file(self):
        """Should handle empty file."""
        types = self.manager.extract_types("", language="csharp")
        assert types == []

    def test_handles_namespace(self):
        """Should handle namespace declaration."""
        source = """
namespace MyApp.Services
{
    public class UserService
    {
    }
}
"""
        types = self.manager.extract_types(source, language="csharp")

        service = next((t for t in types if t.name == "UserService"), None)
        assert service is not None

    def test_handles_file_scoped_namespace(self):
        """Should handle file-scoped namespace."""
        source = """
namespace MyApp.Services;

public class UserService
{
}
"""
        types = self.manager.extract_types(source, language="csharp")
        service = next((t for t in types if t.name == "UserService"), None)

        assert service is not None

    def test_handles_attributes(self):
        """Should handle classes with attributes."""
        source = """
[Serializable]
[Table("users")]
public class User
{
    [Key]
    public int Id { get; set; }
}
"""
        types = self.manager.extract_types(source, language="csharp")

        user = next((t for t in types if t.name == "User"), None)
        assert user is not None

    def test_handles_nullable_types(self):
        """Should handle nullable reference types."""
        source = """
public class Service
{
    public string? GetValue(int? id)
    {
        return null;
    }
}
"""
        methods = self.manager.extract_methods(source, language="csharp")
        get_value = next((m for m in methods if m.name == "GetValue"), None)

        assert get_value is not None
        assert get_value.class_name == "Service"

    def test_handles_partial_class(self):
        """Should handle partial class."""
        source = """
public partial class User
{
    public string Name { get; set; }
}
"""
        types = self.manager.extract_types(source, language="csharp")

        user = next((t for t in types if t.name == "User"), None)
        assert user is not None
